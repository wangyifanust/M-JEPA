#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mem=60G                      
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --qos=debug

# module load python/3.8

source ~/anaconda3/bin/activate mamp


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1


echo "Running on host: $HOSTNAME"


python -m torch.distributed.launch --nproc_per_node=2 --master_port=11234 main_pretrain.py \
--config ./config/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp \
--log_dir ./output_dir/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp
