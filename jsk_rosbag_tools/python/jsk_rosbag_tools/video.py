from __future__ import division

import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile
import wave

import audio_common_msgs.msg
import cv2
import numpy as np
import rosbag
import rospy
from tqdm import tqdm

from jsk_rosbag_tools.cv import img_to_msg
from jsk_rosbag_tools.merge import merge_bag


def mediainfo(filepath):
    prober = 'ffprobe'
    command_args = [
        "-v", "quiet",
        "-show_format",
        "-show_streams",
        filepath
    ]

    command = [prober, '-of', 'old'] + command_args
    res = subprocess.Popen(command, stdout=subprocess.PIPE)
    output = res.communicate()[0].decode("utf-8")

    if res.returncode != 0:
        command = [prober] + command_args
        output = subprocess.Popen(
            command,
            stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    rgx = re.compile(r"(?:(?P<inner_dict>.*?):)?(?P<key>.*?)\=(?P<value>.*?)$")
    info = {}
    if sys.platform == 'win32':
        output = output.replace("\r", "")
    for line in output.split("\n"):
        # print(line)
        mobj = rgx.match(line)

        if mobj:
            # print(mobj.groups())
            inner_dict, key, value = mobj.groups()

            if inner_dict:
                try:
                    info[inner_dict]
                except KeyError:
                    info[inner_dict] = {}
                info[inner_dict][key] = value
            else:
                info[key] = value

    return info


def nsplit(xlst, n):
    total_n = len(xlst)
    d = int((total_n + n - 1) / n)
    i = 0
    ret = []
    while i < total_n:
        ret.append(xlst[i:i + d])
        i += d
    return ret


def load_frame(vid, fps, min_stamp=rospy.Time(0), max_stamp=None,
               compress=False, target_size=None):
    vid_avail = True
    cur_frame = 0
    while True:
        cur_stamp = min_stamp + rospy.Duration(cur_frame / fps)
        vid_avail, frame = vid.read()
        if not vid_avail:
            break
        if max_stamp is not None and cur_stamp > max_stamp:
            break
        if target_size is not None:
            frame = cv2.resize(frame, target_size)
        msg = img_to_msg(frame, compress=compress)
        msg.header.stamp = cur_stamp
        # if cur_frame % 20 == 0:
        #     print(f"{cur_frame} {cur_stamp.to_sec():0.2f}")
        cur_frame += 1
        yield msg


def video_to_bag(video_filepath, bag_output_filepath,
                 topic_name, compress=False, audio_topic_name='/audio',
                 no_audio=False,
                 base_stamp=None,
                 duration_sec=0.0,
                 video_offset_sec=0.0,
                 override_fps=None,
                 show_progress_bar=True):
    if base_stamp is None:
        base_stamp = rospy.Time()

    topic_name = topic_name.lstrip('/compressed')
    if compress is True:
        topic_name = osp.join(topic_name, 'compressed')

    video_filepath = str(video_filepath)
    print(video_filepath)
    vid = cv2.VideoCapture(video_filepath)
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    if override_fps is not None:
        fps = override_fps
        time_scale = override_fps / video_fps
    else:
        fps = video_fps
        time_scale = 1.0
    print(f"video fps is {video_fps:0.6f}, using {fps:0.2f}, time scale {time_scale:0.6f}")
    cur_pos_ms = vid.get(cv2.CAP_PROP_POS_MSEC)
    print(cur_pos_ms)
    vid.set(cv2.CAP_PROP_POS_FRAMES, video_offset_sec * video_fps)
    cur_pos_ms = vid.get(cv2.CAP_PROP_POS_MSEC)
    cur_pos_ind = vid.get(cv2.CAP_PROP_POS_FRAMES)
    print(f"video offset {video_offset_sec}s -> current pos {cur_pos_ms:0.3f}ms, {cur_pos_ind}")

    tmpdirname = tempfile.mkdtemp("", 'tmp', None)
    video_out = osp.join(tmpdirname, 'video.tmp.bag')

    start_offset = video_offset_sec / time_scale
    min_stamp = base_stamp + rospy.Duration(start_offset)
    print(f"base {base_stamp.to_sec():0.3f}s, start {min_stamp.to_sec():0.3f}s")

    n_frame = vid.get(cv2.CAP_PROP_FRAME_COUNT) - cur_pos_ind
    max_stamp = None
    if duration_sec > 0.0:
        max_stamp = min_stamp + rospy.Duration(duration_sec)
        text = f"max stamp {max_stamp.to_sec():0.2f}s"
        text += f" ({duration_sec:0.2f}s {(max_stamp - min_stamp).to_sec():0.2f}s)"
        print(text)
        n_frame = duration_sec * video_fps * time_scale + 1

    # TODO(lucasw) take into account start and duration
    print(f"n_frame {n_frame:0.3f} {type(n_frame)}")
    if show_progress_bar:
        progress = tqdm(total=n_frame)

    with rosbag.Bag(video_out, 'w') as outbag:
        for msg in load_frame(vid, fps,
                              min_stamp=min_stamp,
                              max_stamp=max_stamp,
                              compress=compress):
            outbag.write(topic_name, msg, msg.header.stamp)
            if show_progress_bar:
                progress.update(1)
    if show_progress_bar:
        progress.close()

    extract_audio = True
    if no_audio is False:
        wav_filepath = osp.join(tmpdirname, 'tmp.wav')
        cmd = "ffmpeg -i '{}' '{}'".format(
            video_filepath, wav_filepath)
        proc = subprocess.Popen(cmd, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        proc.wait()

        try:
            wf = wave.open(wav_filepath, mode='rb')
            sample_rate = wf.getframerate()
            wf.rewind()
            buf = wf.readframes(-1)
            if wf.getsampwidth() == 2:
                data = np.frombuffer(buf, dtype='int16')
            elif wf.getsampwidth() == 4:
                data = np.frombuffer(buf, dtype='int32')
            data = data.reshape(-1, wf.getnchannels())
            media_info = mediainfo(wav_filepath)
            print(media_info)
        except RuntimeError:
            extract_audio = False

        if extract_audio:
            rate = 100  # / time_scale
            n = int(np.ceil(data.shape[0] / (sample_rate // rate)))
            print(f"unscaled audio sample rate {sample_rate}, msg rate {rate}, {n} samples")
            channels = data.shape[1]

            audio_out = osp.join(tmpdirname, 'audio.tmp.bag')
            with rosbag.Bag(audio_out, 'w') as outbag:
                audio_info = audio_common_msgs.msg.AudioInfo(
                    channels=channels,
                    sample_rate=int(sample_rate * time_scale),
                    sample_format=media_info['codec_name'].upper(),
                    bitrate=int(media_info['bit_rate']) * int(time_scale),
                    coding_format='wave')
                ros_timestamp = base_stamp
                outbag.write(audio_topic_name + '_info',
                             audio_info, ros_timestamp)
                for i, audio_data in enumerate(nsplit(data, n)):
                    offset_timestamp = rospy.Duration(i / (rate * time_scale))
                    ros_timestamp = base_stamp + offset_timestamp
                    if ros_timestamp < base_stamp:
                        continue
                    msg = audio_common_msgs.msg.AudioData()
                    msg.data = audio_data.reshape(-1).tobytes()
                    # print(f"{i} {ros_timestamp.to_sec()}")
                    if max_stamp is not None and ros_timestamp > max_stamp:
                        break
                    outbag.write(audio_topic_name, msg, ros_timestamp)
            merge_bag(video_out, audio_out, bag_output_filepath)
        else:
            shutil.move(video_out, bag_output_filepath)
    else:
        shutil.move(video_out, bag_output_filepath)

    print(bag_output_filepath)
