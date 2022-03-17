#!/bin/bash

tasks=$1
model=$2
machine_name=$3
niid=$4
client_num=$5
alpha=$6
valid_gpu=$7

machine_name_start="hit"
result=$(echo ${machine_name} | grep "${machine_name_start}")
if [ ${machine_name} = "pclv100-124" ]
then
  run_dir=/raid0/zhuo/code/fednlp/run/fedtlm_alone/fedrun_sweep.py
elif [[ "${result}" == "" ]]
then
  run_dir=/raid1/zhuo/code/fednlp/run/fedtlm_alone/fedrun_sweep.py
else
  run_dir=/data/zhuo/code/fednlp/run/fedtlm_alone/fedrun_sweep.py
fi


python ${run_dir} \
--tasks ${tasks} \
--model_name ${model} \
--machine_name ${machine_name} \
--client_num ${client_num} \
--valid_gpu ${valid_gpu} \
--niid ${niid} \
--alpha ${alpha}
