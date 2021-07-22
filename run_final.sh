#!/bin/bash

# SBATCH --partition=gtx1080
#SBATCH -J run # job name
# SBATCH -o out/job.%j  # stdout location and name %j=job id
# SBATCH -e err/job.%j  # stderr location and name %j=job id
# SBATCH -N 1 #one core
# SBATCH -n 1 #one task
# SBATCH -t 23:30:30  #run time(hh:mm:ss)

# bash search.sh --model-id 4
# bash search.sh --model-id 401
# bash search.sh --model-id 402
# bash search.sh --model-id 4 --label-id 1
# bash search.sh --model-id 4 --label-id 2
# bash search.sh --model-id 4 --label-id 3
# bash search.sh --model-id 4 --label-id 4
#for i in {1..5}
#do
#  bash search.sh --model-id 4 --label-id 0 --repeat-id $i
#done


#for i in {1..5}
#do
#  bash search.sh --model-id 4 --label-id 1 --repeat-id $i
#done


#for i in {1..5}
#do
#  bash search.sh --model-id 4 --label-id 2 --repeat-id $i
#done


#for i in {1..5}
#do
#  bash search.sh --model-id 4 --label-id 3 --repeat-id $i
#done


for i in {1..5}
do
  bash search.sh --model-id 4 -label-id 4 --repeat-id $i
done
