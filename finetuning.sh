#!/bin/bash
#SBATCH --job-name=ntu60
#SBATCH --output=slurm_out2/%j.out
#SBATCH --error=slurm_out2/%j.err
#SBATCH --mem=60G                      
#SBATCH --time=36:20:00
#SBATCH --partition=l40s
#SBATCH --account=vita
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4





# module load python/3.8

source ~/anaconda3/bin/activate mamp


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export OMP_NUM_THREADS=1

echo "Running on host: $HOSTNAME"
# NTU-60 xsub
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xsub_joint/finetune_t120_layer8_decay.yaml \
--output_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/ft_mamp_t120_layer8+5_mask90_tau0.80_ep400_noamp_L2_.1reconv1_mom \
--log_dir /work/vita/datasets/output_dir/ntu60_xsub_joint/ft_mamp_t120_layer8+5_mask90_tau0.80_ep400_noamp_L2_.1reconv1_mom \
--finetune /work/vita/datasets/output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+5_drop0.0_mask90_tau0.80_ep1200_noamp_L2_.1reconv1_mom/checkpoint-1199.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
