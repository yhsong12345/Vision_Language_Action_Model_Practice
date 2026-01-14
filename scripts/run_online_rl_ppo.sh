#!/bin/bash
#SBATCH --job-name=vla_ppo
#SBATCH --comment="VLA Online RL - PPO (Proximal Policy Optimization)"
#SBATCH --nodelist=cubox01,cubox02,cubox03,cubox04,cubox06,cubox07,cubox10,cubox11
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=8G

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

# Huggingface cache directory
export HF_HOME=./cache/hf

huggingface-cli login --token $HF_TOKEN

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
echo "World size: $NUM_NODES x $GPUS_PER_NODE = $WORLD_SIZE"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "$nodes"

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_port=29500

# NCCL network configuration (for Infiniband)
export NCCL_SOCKET_IFNAME=eno1

echo "****Starting HEAD at $head_node, $head_node_ip:$head_port"

# Single node, 8 processes
srun --nodes=1 --ntasks=1 -w "$head_node" \
    accelerate launch \
        --config_file config/deepspeed_zero3.yaml \
        --num_machines "$NUM_NODES" \
        --num_processes "$WORLD_SIZE" \
        --main_process_ip "$head_node_ip" \
        --main_process_port "$head_port" \
        --machine_rank 0 \
    train/online_rl/ppo_trainer.py "$@" &
sleep 15

# Start worker from 1 (0 is head node)
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "****Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
      accelerate launch \
        --config_file config/deepspeed_zero3.yaml \
        --num_machines "$NUM_NODES" \
        --num_processes "$WORLD_SIZE" \
        --main_process_ip "$head_node_ip" \
        --main_process_port "$head_port" \
        --machine_rank "$i" \
      train/online_rl/ppo_trainer.py "$@" &
    sleep 5
done

wait
