---
title: HPC tutorial for thesis students
tags: [hpc, slurm, thesis]
updated: 2026-07-16
---

# HPC tutorial for thesis students

## How to connect

To connect to the HPC use the command:
```bash
ssh <user>@hpc-legionlogin.polito.it
```
## How to use SLURM
To launch your experiment you need the following things:
- a built apptainer/singularity container (`.sif`)
- a `.sbatch` file
- all the files needed to run the experiment
### SIF
The container should be built outside of the HPC and then transferred there (eg. with `scp`). It should be built such that it can run your experiment immediately with the `apptainer exec` command.
### SBATCH
This is a SLURM batch script used to submit a job to the cluster. Each `#SBATCH` directive specifies a job configuration parameter. Below is a detailed explanation of each parameter:
```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=serics_gpu
#SBATCH --job-name=openspeech_test
#SBATCH --output=./slurm_outputs/OPENSPEECH.out
#SBATCH --error=./slurm_outputs/OPENSPEECH.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G

apptainer exec --nv container.sif python3 launch.py
```
- **`ntasks`** – the number of tasks to launch. In this case, only one task will be launched. For most single-GPU experiments, `ntasks=1` is appropriate.
- **`partition`** – specifies the partition (i.e., queue) to submit the job to. Available options:
    - `serics_gpu`: our GPU nodes without wall time limits
    - `serics_cpu`: our CPU nodes without wall time limits
    - `cpu_sapphire`, `gpu_a40`: global partitions with 24-hour wall time
    - `cpu_sapphire_ext`, `gpu_a40_ext`, `gpu_a100`: extended public partitions with a 5-day wall time
    - TBD: other partitions exist (e.g. `gpu_v100`, `gpu_v100_ext`, `gpu_h200`). Explore them yourself with `sinfo`
- **`job-name`** – sets the name of the job. Use **descriptive and meaningful names** so you can easily identify the job later (e.g., `pretrain_cifar10`, `finetune_libri100`).
- **`output`** and **`error`** – standard output and error streams will be written to the specified files. Use these logs to monitor and debug your job.
- **`gres=gpu:1`** – requests 1 GPU for the job. Adjust this number if your experiment requires more GPUs. GPU nodes have 4 GPUs.
- **`cpus-per-task=12`** – requests N CPU cores for the job. Request only one GPU's share of the node (on our A40 nodes: 12 cores per GPU); asking for more leaves the node's other GPUs idle and makes your job start later.
- **Nodes** – you don't need to pick one, SLURM assigns it automatically. Nodes on our partitions:
    - GPU NODES: `compute-4-12`, `compute-5-11`, `compute-5-14`
    - CPU NODES: `compute-7-2`, `compute-7-3`
- **`mem`** – requests N GB of system memory. Request only one GPU's share of the node. Max per node is 256 GB.
Lastly you need to insert the command launched by the job that should have the following structure (`--nv` is optional, use it if you need the GPU):
```bash
apptainer exec --nv <container_you_built.sif> <command_to_launch_experiment>
```
### Files needed
This includes scripts, datasets, repo, etc. You can clone a repository in your directory or send one with `scp`.
## Useful SLURM commands
To **launch** a job:
```bash
sbatch <file.sbatch>
```
To see **running** jobs:
```bash
squeue # see all the running jobs on the HPC
squeue -u <user> # see all the job launched from the smilies account
squeue | grep -i <node-name> # see if the node is already occupied with another job
```
To **cancel** a job:
```bash
scancel <job_id>
```
## Things to keep in mind
1. It's a shared environment, don't launch hundreds of jobs at the same time, and be mindful of shared resources.
2. Remember to be sure that your code can actually work on multiple nodes/GPUS/CPUS if you use them to avoid waisting resources.
## Profiling
If you are running an experiment on a node you can see its CPU and GPU usage by connecting a shell to the node with:
```bash
srun --pty --overlap --jobid <job-id of your experiment> bash
```
And then launch the `top`, `htop` or `nvidia-smi` commands.

For more informations and rules visit: https://hpc.polito.it

## Connection Problems
The HPC can be accessed only through the PoliTO's network. You can either use the university's VPN or go through our server. To do that look into `ProxyJump` in the ssh config file, eg:
```
Host hpc
  HostName hpc-legionlogin.polito.it
  User <user>
  ProxyJump daredevil 

Host daredevil
  HostName daredevil.polito.it
  User <user>
```