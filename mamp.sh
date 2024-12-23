#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out14/%j.out
#SBATCH --error=slurm_out14/%j.err
#SBATCH --mem=100G                      
#SBATCH --time=20:20:00
#SBATCH --partition=h100

#SBATCH --cpus-per-task=40
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --mail-user=yifan.wang@epfl.ch

# module load python/3.8

source ~/anaconda3/bin/activate mamp


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
OMP_NUM_THREADS=4

echo "Running on host: $HOSTNAME"


nvidia-smi
nvcc --version


torchrun --nproc_per_node=4   --master_port 11234 main_pretrain.py \
--config ./config/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90.yaml \
--output_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_drop0.0_mask90_tau0.80_ep600_noamp_Linear_MSE_archCv4 \
--log_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_drop0.0_mask90_tau0.80_ep600_noamp_Linear_MSE_archCv4 \

