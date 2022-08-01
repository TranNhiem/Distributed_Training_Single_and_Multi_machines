# SLURM SUBMIT SCRIPT

#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-02:00:00

# (optional) debugging flags
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# (optional) set the network interface name manually
# export NCCL_SOCKET_IFNAME=lo

# run full training with ImageNet data already downloaded
srun python train.py --trainer.num_nodes 2 --trainer.gpus 8 \
--data_path path/to/dataset
