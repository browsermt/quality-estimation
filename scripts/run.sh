#!/bin/bash

out_pref=$1

echo "#################################################################"
echo "## This script fits feature-based qe model for enes enet and encs"
echo "## This models will be saved to $out_pref.<langpair>.json"
echo "#################################################################"
echo

for l in es cs et ; do
  if [ $l == "cs" ]
  then
    python quality_estimation/train.py --path_train_src data/wiki.en$l.src --path_train_mt data/wiki.en$l.mt.plain \
    --path_train_mt_info data/wiki.en$l.mt.probs --path_train_labels data/wiki.en$l.labels \
    --save_model_path $out_pref.en$l.json --retain_eos
  else
    python quality_estimation/train.py --path_train_src data/wiki.en$l.src --path_train_mt data/wiki.en$l.mt.plain \
    --path_train_mt_info data/wiki.en$l.mt.probs --path_train_labels data/wiki.en$l.labels \
    --save_model_path $out_pref.en$l.json
  fi
done
