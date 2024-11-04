#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mem=60G                      
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4




# module load python/3.8

source ~/anaconda3/bin/activate mamp


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export OMP_NUM_THREADS=1

echo "Running on host: $HOSTNAME"

# NTU-60 xsub
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
# --config ./config/ntu60_xsub_joint/linprobe_t120_layer8.yaml \
# --output_dir ./output_dir/ntu60_xsub_joint/linear_mae_t120_layer8+3_mask90_ep400_400 \
# --log_dir ./output_dir/ntu60_xsub_joint/linear_mae_t120_layer8+3_mask90_ep400_400 \
# --finetune ./output_dir/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp/checkpoint-399.pth \
# --dist_eval
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main_linprobe.py \
--config ./config/ntu60_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/linear_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp_L2v1+Con0.1v1 \
--log_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/linear_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp_L2v1+Con0.1v1 \
--finetune /home/yifwang/M-JEPA/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp_L2+Con0.1v1/checkpoint-380.pth \
--dist_eval