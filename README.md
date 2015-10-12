jsk_common
===

[![Build Status](https://travis-ci.org/jsk-ros-pkg/jsk_common.svg?branch=master)](https://travis-ci.org/jsk-ros-pkg/jsk_common)
[![Slack](https://img.shields.io/badge/slack-jsk--robotics-e100e1.svg)](http://jsk-robotics.slack.com)
[![Join the chat at https://gitter.im/jsk-ros-pkg/jsk_common](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/jsk-ros-pkg/jsk_common?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Install
---
You can use `jsk.rosbuild` to setup your environment.


```sh
wget -q -O /tmp/jsk.rosbuild https://raw.github.com/jsk-ros-pkg/jsk_common/master/jsk.rosbuild
bash /tmp/jsk.rosbuild hydro
```

For hacker

```sh
wget -q -O /tmp/jsk.rosbuild https://raw.github.com/jsk-ros-pkg/jsk_common/master/jsk.rosbuild
bash /tmp/jsk.rosbuild --from-source hydro
```

For hrpsys user

```sh
wget -q -O /tmp/jsk.rosbuild https://raw.github.com/jsk-ros-pkg/jsk_common/master/jsk.rosbuild
bash /tmp/jsk.rosbuild --rtm hydro
```

For hrpsys hacker

```sh
wget -q -O /tmp/jsk.rosbuild https://raw.github.com/jsk-ros-pkg/jsk_common/master/jsk.rosbuild
bash /tmp/jsk.rosbuild --from-source --rtm hydro
```

`jsk.rosbuild` generates filesystem as follows:

```
~ --- ros
       + --- hydro_parent: Only availabe if --from-source option is enabled
              + --- src:   maintained by wstool
              + --- build: generated by catkin_tools
              + --- devel: generated by catkin_tools
       +--- hydro
             + --- src:    maintained by wstool
             + --- build:  generated by catkin_tools
             + --- devel:  generated by catkin_tools
```

Watch all the jsk github repositories.
===
Please use [this](http://jsk-github-watcher.herokuapp.com/)

Slack for JSK Lab members ![](https://upload.wikimedia.org/wikipedia/en/7/76/Slack_Icon.png)
=========================
You can login to [slack](https://slack.com/) from [here](https://jsk-robotics.slack.com).
You can create account using imi address.

[scudcloud](https://github.com/raelgc/scudcloud) is a desktop client for slack and you can install it
by following [instruction](https://github.com/raelgc/scudcloud#ubuntukubuntu-mint-and-debian).

You can restart travis and jenkins from slack's `#travis` channel.

Restart travis from slack
-------------------------
![](images/restart_travis.png)

Type `restart travis <job-id>` from slack#travis channel.

**N.B.: `<job-id>` is not the number of Pull-request.**

you can get `<job-id>` from Travis page.

- ![](images/PR_page.png)
- ![](images/Travis_page.png)

Restart docker from slack
-------------------------
![](images/restart_docker.png)

Type `restart docker` from slack#travis channel.


Deb Build Status
-----------------

hydro

- sourcedeb [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-hydro-jsk-common_sourcedeb)](http://jenkins.ros.org/job/ros-hydro-jsk-common_sourcedeb/)
- binarydeb precise amd64 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-hydro-jsk-common_binarydeb_precise_amd64)](http://jenkins.ros.org/job/ros-hydro-jsk-common_binarydeb_precise_amd64/)
- binarydeb precise i386 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-hydro-jsk-common_binarydeb_precise_i386)](http://jenkins.ros.org/job/ros-hydro-jsk-common_binarydeb_precise_i386/)
- binarydeb quantal amd64 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-hydro-jsk-common_binarydeb_quantal_amd64)](http://jenkins.ros.org/job/ros-hydro-jsk-common_binarydeb_quantal_amd64/)
- binarydeb quantal i386 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-hydro-jsk-common_binarydeb_quantal_i386)](http://jenkins.ros.org/job/ros-hydro-jsk-common_binarydeb_quantal_i386/)
- binarydeb raring amd64 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-hydro-jsk-common_binarydeb_raring_amd64)](http://jenkins.ros.org/job/ros-hydro-jsk-common_binarydeb_raring_amd64/)
- binarydeb raring i386 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-hydro-jsk-common_binarydeb_raring_i386)](http://jenkins.ros.org/job/ros-hydro-jsk-common_binarydeb_raring_i386/)

indigo
- sourcedeb [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-indigo-jsk-common_sourcedeb)](http://jenkins.ros.org/job/ros-indigo-jsk-common_sourcedeb/)
- binarydeb saucy amd64 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-indigo-jsk-common_binarydeb_saucy_amd64)](http://jenkins.ros.org/job/ros-indigo-jsk-common_binarydeb_saucy_amd64/)
- binarydeb saucy i386 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-indigo-jsk-common_binarydeb_saucy_i386)](http://jenkins.ros.org/job/ros-indigo-jsk-common_binarydeb_saucy_i386/)
- binarydeb trusty amd64 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-indigo-jsk-common_binarydeb_trusty_amd64)](http://jenkins.ros.org/job/ros-indigo-jsk-common_binarydeb_trusty_amd64/)
- binarydeb trusty i386 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-indigo-jsk-common_binarydeb_trusty_i386)](http://jenkins.ros.org/job/ros-indigo-jsk-common_binarydeb_trusty_i386/)

jade
- sourcedeb [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-jade-jsk-common_sourcedeb)](http://jenkins.ros.org/job/ros-jade-jsk-common_sourcedeb/)
- binarydeb trusty amd64 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-jade-jsk-common_binarydeb_trusty_amd64)](http://jenkins.ros.org/job/ros-jade-jsk-common_binarydeb_trusty_amd64/)
- binarydeb trusty i386 [![Build Status](http://jenkins.ros.org/buildStatus/icon?job=ros-jade-jsk-common_binarydeb_trusty_i386)](http://jenkins.ros.org/job/ros-jade-jsk-common_binarydeb_trusty_i386/)
