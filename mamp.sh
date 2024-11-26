#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out12/%j.out
#SBATCH --error=slurm_out12/%j.err
#SBATCH --mem=90G                      
#SBATCH --time=23:20:00
#SBATCH --partition=h100

#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4


# module load python/3.8

source ~/anaconda3/bin/activate mamp


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
OMP_NUM_THREADS=1

echo "Running on host: $HOSTNAME"


torchrun --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90.yaml \
--output_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp_L2v8 \
--log_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp_L2v8

