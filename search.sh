#!/bin/bash

# SBATCH --partition=gtx1080
# SBATCH -J run # job name


set -e
set +o posix

num_epoch=10
model_id=4
label_id=1
repeat_id=1
. ./utils/parse_options.sh

python3 train.py \
  --only-single-label ${label_id} \
  --num-epochs ${num_epoch} \
  --model-select $model_id \
  --opath2history "history/search.pickle" \
  --opath2model 'model/singlelabel_search.hdf5'
 
python3 test.py \
  --only-single-label ${label_id} \
  --ipath2model "model/singlelabel_search.hdf5" \
  --log-filename "log/model_${model_id}_label_${label_id}_repeat_${repeat_id}.log"

echo "result is stored in log/model_${model_id}_label_${label_id}_repeat_${repeat_id}.log"
