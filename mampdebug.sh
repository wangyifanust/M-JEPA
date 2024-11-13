#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out4/%j.out
#SBATCH --error=slurm_out4/%j.err
#SBATCH --mem=80G                      
#SBATCH --time=47:20:00
#SBATCH --partition=l40s
#SBATCH --account=vita
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=2
#SBATCH --gres=gpu:4


# module load python/3.8

source ~/anaconda3/bin/activate mamp


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
OMP_NUM_THREADS=1

echo "Running on host: $HOSTNAME"



python -m torch.distributed.launch --nproc_per_node=4   --master_port 11234 main_pretrain.py \
--config ./config/ntu60_xsub_joint/pretrain_mamp_t120_layer8+5_mask90.yaml \
--output_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+5_drop0.0_mask90_tau0.80_ep1200_noamp_L2_.00reconv1_momv2 \
--log_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+5_drop0.0_mask90_tau0.80_ep1200_noamp_L2_.00reconv1_momv2 \

