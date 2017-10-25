#!/usr/bin/env bash
# download one video sample (UCF-101), extract frames, and prepare for
# deepflow (making a list of successive frame pairs)

VIDEO=v_CricketShot_g04_c01

# retrieve video
wget -nc http://crcv.ucf.edu/THUMOS14/UCF101/UCF101/${VIDEO}.avi

# extract frames
mkdir -p frames
# potentially remove all previous files
rm -f frames/[0-9][0-9][0-9][0-9][0-9][0-9].png
ffmpeg \
  -i ${VIDEO}.avi \
  frames/%06d.png

# generate an input file (a list of image pairs) for flownet2
FILES=$(ls frames/*.png | sort)
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FLOW_LIST=frames/flow_list.txt
PREV_FILE=""
rm -f ${FLOW_LIST}
for FILE in ${FILES}; do
  echo Processing ${FILE}...
  if [[ ! -z ${PREV_FILE} ]]; then
    FLOW_FILE=${PREV_FILE/.png/.flo}
    echo "${CWD}/${PREV_FILE} ${CWD}/${FILE} ${CWD}/${FLOW_FILE}" >> ${FLOW_LIST}
  fi
  PREV_FILE=${FILE}
done;

# now, run flownet script to generate flow files
