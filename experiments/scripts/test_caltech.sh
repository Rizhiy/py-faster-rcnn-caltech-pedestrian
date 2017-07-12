#!/usr/bin/env bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh [options args to test_net.py]
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh RNG_SEED 42

set +x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=VGG16

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB="caltech_train_1x"
TEST_IMDB="caltech_test_1x"
PT_DIR="caltech"
ITERS=1000

LOG="experiments/logs/test_caltech_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

NET_FINAL=/home/rizhiy/pedestrian_detection/rcnn/py-faster-rcnn-caltech-pedestrian/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel

time ./tools/test_net.py \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
