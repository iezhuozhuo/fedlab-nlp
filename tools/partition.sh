#!/bin/bash

dataset=$1
machine_name=$2
niid=$3
client_num=$4
alpha=$5

machine_name_start="hit"
result=$(echo ${machine_name} | grep "${machine_name_start}")

if [ ${machine_name} = "pclv100-124" ]
then
  run_dir=/raid0
elif [ ${machine_name} = "uestc" ]
then
  run_dir=/hdd
elif [ "${result}" == "" ]
then
  run_dir=/raid1
else
  run_dir=/data
fi

if [ ${niid} == "0" ]
then
  python ${run_dir}/zhuo/code/fednlp/data/advanced_partition/niid_label.py \
  --client_number ${client_num} \
  --data_file ${run_dir}/zhuo/data/fednlp_data/data_files/${dataset}_data.h5 \
  --partition_file ${run_dir}/zhuo/data/fednlp_data/partition_files/${dataset}_partition.h5 \
  --task_type text_classification \
  --seed 0 \
  --alpha ${alpha} \
  --skew_type label
else
  python ${run_dir}/zhuo/code/fednlp/data/advanced_partition/iid_label.py \
  --client_number ${client_num} \
  --data_file ${run_dir}/zhuo/data/fednlp_data/data_files/${dataset}_data.h5 \
  --partition_file ${run_dir}/zhuo/data/fednlp_data/partition_files/${dataset}_partition.h5
fi
