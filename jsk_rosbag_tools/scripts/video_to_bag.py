#!/usr/bin/env python

import argparse
import os.path as osp

import rospy
from jsk_rosbag_tools.video import video_to_bag


def main():
    parser = argparse.ArgumentParser(
        description='Convert video to bag.')
    parser.add_argument('inputvideo')
    parser.add_argument('--out', '-o', type=str,
                        help='name of the output bag file',
                        default=None, metavar="output_file")
    parser.add_argument('--topic-name', type=str,
                        default='/video/rgb/image_raw',
                        help='Converted topic name.')
    parser.add_argument('--fps', type=float,
                        help='Frame Rate.', default=None)
    parser.add_argument('--time_offset_sec', type=float,
                        help='Unix epoch time offset.', default=0)
    parser.add_argument('--compress', action='store_true',
                        help='Compress Image flag.')
    parser.add_argument('--no-progress-bar', action='store_true',
                        help="Don't show progress bar.")
    args = parser.parse_args()

    base_stamp = rospy.Time.from_sec(args.time_offset_sec)
    print(f"{args.time_offset_sec} -> {base_stamp.to_sec()}")

    video_path = args.inputvideo
    if args.out is None:
        args.out = osp.join(
            osp.dirname(video_path),
            osp.splitext(osp.basename(video_path))[0] + '.bag')

    outfile = args.out
    pattern = str(osp.join(
        osp.dirname(video_path),
        osp.splitext(osp.basename(video_path))[0] + "_%i.bag"))
    index = 0
    while osp.exists(outfile):
        outfile = pattern % index
        index += 1
    video_to_bag(
        video_path, outfile,
        args.topic_name,
        compress=args.compress,
        no_audio=False,  # TODO(lucasw) args
        base_stamp=base_stamp,
        override_fps=args.fps,
        show_progress_bar=not args.no_progress_bar)


if __name__ == '__main__':
    main()
