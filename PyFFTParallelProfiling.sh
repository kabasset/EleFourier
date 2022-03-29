#!/bin/sh

# SLURM options:

#SBATCH --job-name=PyFFTParallelProfiling    # Job name
#SBATCH --output=PyFFTParallelProfiling_%j.log   # Standard output and error log

#SBATCH --partition flash              # Partition choice
#SBATCH --ntasks 20                    # Run a single task (by default tasks == CPU)

#SBATCH --mail-user=manuel.grizonnet@cnes.fr  # Where to send mail
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

source /cvmfs/euclid-dev.in2p3.fr/CentOS7/EDEN-3.0/bin/activate

cd /pbs/home/g/grizonnm/Work/Projects/EleFourier/
END=30
for i in $(seq 1 $END); do time ./build.x86_64-conda_cos6-gcc93-o2g/run PyFFTParallel -l 40 -p 20 -b $i; done
