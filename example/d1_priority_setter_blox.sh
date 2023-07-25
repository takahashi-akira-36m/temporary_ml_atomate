#!/bin/zsh
#$ -S /bin/zsh
#$ -cwd
#$ -V
#$ -j y
#$ -o priority.log
#$ -pe all_pe* 36

ml_atomate_path=${HOME}/ml_atomate
python ${ml_atomate_path}/ml_atomate/priority_setter.py -df ~/fireworks_config_atomate/db.json -bld ${ml_atomate_path}/example/atomate_files/run_builder.py -dc descriptors.csv --objective bandstructure_hse.bandgap dielectric.epsilon_avg -rs 0 -ad -nrb 0 -np 20 -c no_conversion log --blox


