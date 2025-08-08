#!/bin/bash
#SBATCH -A p30041
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8GB
#SBATCH -t 2:00:00
#SBATCH --job-name="retro"
#SBATCH --output=/home/spn1560/krxns/logs/out/%A
#SBATCH --error=/home/spn1560/krxns/logs/error/%A
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/krxns/scripts/retrosynthesis.py
expansion=3_steps_lbnl_targets_to_None_rules_mechinferred_dt_04_rules_w_coreactants_co_metacyc_coreactants_sampled_False_pruned_False_aplusb_False
max_depth=5
max_leaves=3

# Commands
ulimit -c 0
module purge
source /home/spn1560/krxns/.venv/bin/activate
python $script max_depth=$max_depth max_leaves=$max_leaves expansion=$expansion