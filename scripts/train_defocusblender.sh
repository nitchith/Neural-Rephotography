#!/bin/bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script for training on the Blender dataset.

SCENE=lego
# EXPERIMENT=train_4stacks
# DATA_DIR=/home/srinitca/capstone/nerf-pytorch/data/nerf_newdataset_4stack/

EXPERIMENT=train_1stacks
DATA_DIR=/home/srinitca/capstone/nerf-pytorch/data/nerf_newdataset_1stack/
# DATA_DIR=/home/srinitca/capstone/nerf-pytorch/data/nerf_newview/

TRAIN_DIR=/home/srinitca/capstone/NeReFocus/tmp_defocus_results/$EXPERIMENT/$SCENE
#DATA_DIR=/home/srinitca/capstone/dataset/nerf_synthetic/$SCENE/

export LD_LIBRARY_PATH=/home/srinitca/capstone/NeReFocus/libs:$LD_LIBRARY_PATH

#rm -rf $TRAIN_DIR/
python -m train \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --gin_file=configs/defocusblender.gin \
  --logtostderr \
  --gin_param="Config.batch_size=512"
