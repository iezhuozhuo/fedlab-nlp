#!/bin/bash

dataset=$1
machine_name=$2

machine_name_start="hit"
result=$(echo ${machine_name} | grep "${machine_name_start}")

if [ ${machine_name} = "pclv100-124" ]
then
  run_dir=/raid2
elif [ ${machine_name} = "uestc" ]
then
  run_dir=/hdd
elif [ "${result}" == "" ]
then
  run_dir=/ghome
else
  run_dir=/data
fi
# visualization_heatmap_label
#--partition_name niid_label_clients=10_alpha=10.0 \
# visualization_distplot
# visualization_heatmap
# visualization_heatmap_unsort
python ${run_dir}/zhuo/code/fednlp/data/advanced_partition/util/visualization_heatmap_unsort.py \
--partition_name niid_label_clients=10_alpha=10.0 \
--partition_file ${run_dir}/zhuo/data/fednlp_data/partition_files/niid_${dataset}_pdata.h5 \
--data_file ${run_dir}/zhuo/data/fednlp_data/data_files/${dataset}_data.h5 \
--task_name ${dataset} \
--task_type text_classification \
--client_number 10 \
--figure_path ${run_dir}/zhuo/output/fednlp/figure/