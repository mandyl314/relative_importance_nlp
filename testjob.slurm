#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=130:00:00          # total run time limit (HH:MM:SS)
#SBATCH -o /scratch/gpfs/mandyl/relative_importance/ml_slurm-%j.out
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mandyl@princeton.edu

module purge
module load anaconda3/2020.11
source nlpenv/bin/activate


cd /scratch/gpfs/mandyl/relative_importance/


# pip3 install -r requirements.txt
python -m spacy download en_core_web_md
# python analyze_all.py


#python extract_all.py
#python extract_all_attention_ablation.py
#python analyze_all_attention_ablation.py
python test_gpt2.py

