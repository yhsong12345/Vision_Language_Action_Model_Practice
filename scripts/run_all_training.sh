#!/bin/bash
#
# VLA Complete Training Pipeline
# Runs all training stages in sequence or parallel
#
# Usage:
#   ./run_all_training.sh [stage] [options]
#
# Stages:
#   pretrain    - VLM pretraining (vision-language alignment)
#   il          - Imitation learning (BC/DAgger/GAIL)
#   online_rl   - Online RL (PPO/SAC/GRPO)
#   offline_rl  - Offline RL (CQL/IQL/TD3+BC/DT)
#   world_model - World model training
#   driving     - Autonomous driving VLA
#   humanoid    - Humanoid robot VLA
#   all         - Run all stages sequentially
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default stage
STAGE=${1:-"all"}
shift || true

echo "=========================================="
echo "VLA Training Pipeline"
echo "Stage: $STAGE"
echo "Project Root: $PROJECT_ROOT"
echo "=========================================="

# Function to submit a SLURM job
submit_job() {
    local script=$1
    local job_name=$2
    shift 2

    echo "Submitting job: $job_name"
    sbatch "$SCRIPT_DIR/$script" "$@"
}

# Function to wait for job completion
wait_for_job() {
    local job_id=$1
    echo "Waiting for job $job_id to complete..."
    while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
        sleep 60
    done
    echo "Job $job_id completed"
}

case "$STAGE" in
    "pretrain")
        echo "Starting VLM Pretraining..."
        submit_job "run_pretrain.sh" "vla_pretrain" "$@"
        ;;

    "il")
        echo "Starting Imitation Learning..."
        submit_job "run_il.sh" "vla_il" "$@"
        ;;

    "online_rl")
        echo "Starting Online RL Training..."
        echo "Select algorithm (ppo/sac/grpo):"
        read -r algo
        case "$algo" in
            "ppo") submit_job "run_online_rl_ppo.sh" "vla_ppo" "$@" ;;
            "sac") submit_job "run_online_rl_sac.sh" "vla_sac" "$@" ;;
            "grpo") submit_job "run_online_rl_grpo.sh" "vla_grpo" "$@" ;;
            *) echo "Unknown algorithm: $algo"; exit 1 ;;
        esac
        ;;

    "offline_rl")
        echo "Starting Offline RL Training..."
        echo "Select algorithm (cql/iql/td3bc/dt):"
        read -r algo
        case "$algo" in
            "cql") submit_job "run_offline_rl_cql.sh" "vla_cql" "$@" ;;
            "iql") submit_job "run_offline_rl_iql.sh" "vla_iql" "$@" ;;
            "td3bc") submit_job "run_offline_rl_td3bc.sh" "vla_td3bc" "$@" ;;
            "dt") submit_job "run_offline_rl_dt.sh" "vla_dt" "$@" ;;
            *) echo "Unknown algorithm: $algo"; exit 1 ;;
        esac
        ;;

    "world_model")
        echo "Starting World Model Training..."
        submit_job "run_world_model.sh" "vla_world_model" "$@"
        ;;

    "driving")
        echo "Starting Autonomous Driving VLA Training..."
        submit_job "run_driving_vla.sh" "vla_driving" "$@"
        ;;

    "humanoid")
        echo "Starting Humanoid Robot VLA Training..."
        submit_job "run_humanoid_vla.sh" "vla_humanoid" "$@"
        ;;

    "all")
        echo "Running complete training pipeline..."

        # Stage 1: Pretraining
        echo "[1/7] VLM Pretraining..."
        JOB1=$(sbatch --parsable "$SCRIPT_DIR/run_pretrain.sh" "$@")
        wait_for_job "$JOB1"

        # Stage 2: Imitation Learning
        echo "[2/7] Imitation Learning..."
        JOB2=$(sbatch --parsable "$SCRIPT_DIR/run_il.sh" "$@")
        wait_for_job "$JOB2"

        # Stage 3: World Model
        echo "[3/7] World Model Training..."
        JOB3=$(sbatch --parsable "$SCRIPT_DIR/run_world_model.sh" "$@")
        wait_for_job "$JOB3"

        # Stage 4: Offline RL (can run multiple in parallel)
        echo "[4/7] Offline RL Training..."
        JOB4=$(sbatch --parsable "$SCRIPT_DIR/run_offline_rl_iql.sh" "$@")
        wait_for_job "$JOB4"

        # Stage 5: Online RL Fine-tuning
        echo "[5/7] Online RL Fine-tuning..."
        JOB5=$(sbatch --parsable "$SCRIPT_DIR/run_online_rl_grpo.sh" "$@")
        wait_for_job "$JOB5"

        # Stage 6: Driving VLA (if applicable)
        echo "[6/7] Driving VLA Training..."
        JOB6=$(sbatch --parsable "$SCRIPT_DIR/run_driving_vla.sh" "$@")
        wait_for_job "$JOB6"

        # Stage 7: Humanoid VLA (if applicable)
        echo "[7/7] Humanoid VLA Training..."
        JOB7=$(sbatch --parsable "$SCRIPT_DIR/run_humanoid_vla.sh" "$@")
        wait_for_job "$JOB7"

        echo "=========================================="
        echo "Complete training pipeline finished!"
        echo "=========================================="
        ;;

    *)
        echo "Unknown stage: $STAGE"
        echo "Available stages: pretrain, il, online_rl, offline_rl, world_model, driving, humanoid, all"
        exit 1
        ;;
esac

echo "Done!"
