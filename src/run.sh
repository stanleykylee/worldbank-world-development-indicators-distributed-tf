#!/bin/bash

python trainer.py \
     --ps_hosts=52.202.111.255:2222 \
     --worker_hosts=54.196.220.75:2222,52.70.236.58:2222,34.201.244.78:2222 \
     --job_name=ps --task_index=0

python trainer.py \
     --ps_hosts=52.202.111.255:2222 \
     --worker_hosts=54.196.220.75:2222,52.70.236.58:2222,34.201.244.78:2222 \
     --job_name=worker --task_index=0

python trainer.py \
     --ps_hosts=52.202.111.255:2222 \
     --worker_hosts=54.196.220.75:2222,52.70.236.58:2222,34.201.244.78:2222 \
     --job_name=worker --task_index=1

python trainer.py \
     --ps_hosts=52.202.111.255:2222 \
     --worker_hosts=54.196.220.75:2222,52.70.236.58:2222,34.201.244.78:2222 \
     --job_name=worker --task_index=2
