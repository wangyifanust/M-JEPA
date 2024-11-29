#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out9/%j.out
#SBATCH --error=slurm_out9/%j.err
#SBATCH --mem=60G                      
#SBATCH --time=30:20:00
#SBATCH --partition=h100
#SBATCH --account=vita
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3


# module load python/3.8

source ~/anaconda3/bin/activate mamp


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
OMP_NUM_THREADS=4

echo "Running on host: $HOSTNAME"



torchrun --nproc_per_node=3   --master_port 11234 main_pretrain.py \
--config ./config/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90_archB.yaml \
--output_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_drop0.0_mask90_tau0.80_ep600_noamp_MLP_MSE_archB_v4 \
--log_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_drop0.0_mask90_tau0.80_ep600_noamp_MLP_MSE_archB_v4 \

