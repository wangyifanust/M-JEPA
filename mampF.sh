#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out16/%j.out
#SBATCH --error=slurm_out16/%j.err
#SBATCH --mem=30G                      
#SBATCH --time=40:20:00
#SBATCH --partition=gpu

#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2


# module load python/3.8

source ~/anaconda3/bin/activate mamp


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
OMP_NUM_THREADS=4

echo "Running on host: $HOSTNAME"



torchrun --nproc_per_node=2   --master_port 11234 main_pretrain.py \
--config ./config/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90_archF.yaml \
--output_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_drop0.0_mask80_tau0.80_ep600_noamp_INv2_MSE_archF_ema_0.999reg \
--log_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_drop0.0_mask80_tau0.80_ep600_INv2_Linear_MSE_archF_ema_0.999reg \

