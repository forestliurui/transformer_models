#!/bin/bash
#
#BSUB -a openmpi            		# tell lsf this is an openmpi job
#BSUB -W 120:0               		# total wall-clock time (120 h = 5 days)
#BSUB -n 1                    		# number of tasks in job
#BSUB -R "span[ptile=1]" 		# limit 20 processes per node. See note above about HT
#BSUB -R rusage[mem=10000] 	# amount of total memory in MB for all processes
#BSUB -R "affinity[core(10):cpubind=core:distribute=balance]"
#BSUB -R "select[hname=='gpu-cn008']" # gpu-cn001 always give me cuda OOM error
#BSUB -M 10000			# amount of memory in MB per process
#BSUB -J electra_fine_tuning			# job name
#BSUB -e errors.%J      		# error file name in which %J is replaced by the job ID
#BSUB -o output.%J      		# output file name in which %J is replaced by the job ID
#BSUB -q gpu_p100			# choose the queue to use: see list below
#BSUB -B 				# email job start notification
#BSUB -N 				# email job end notification
#BSUB -u ruixliu@umich.edu	# email address to send notifications

cd ../PyTorch

CUDA_VISIBLE_DEVICES=2,3 python run_finetuning_glue.py --model=bert --config_file="bert-base-single-node-rui.json" --config_file_path="/gpfs/gpfs0/groups/mozafari/ruixliu/code/distributed_learning/pytorch/src/bert/pretrain/configs" --data_dir=/gpfs/gpfs0/groups/mozafari/ruixliu/data/glue_data/SST-2 --task_name=sst2 --output_dir=/gpfs/gpfs0/groups/mozafari/ruixliu/tmp/output --do_train --do_eval --do_lower_case --checkpoint_file=/gpfs/gpfs0/groups/mozafari/ruixliu/tmp/models/20200307_104151_0_bert_encoder_epoch_0001.pt

CUDA_VISIBLE_DEVICES=2,3 python run_finetuning_glue.py --model=bert --config_file="bert-base-single-node-rui.json" --config_file_path="/gpfs/gpfs0/groups/mozafari/ruixliu/code/distributed_learning/pytorch/src/bert/pretrain/configs" --data_dir=/gpfs/gpfs0/groups/mozafari/ruixliu/data/glue_data/STS-B --task_name=stsb --output_dir=/gpfs/gpfs0/groups/mozafari/ruixliu/tmp/output --do_train --do_eval --do_lower_case --checkpoint_file=/gpfs/gpfs0/groups/mozafari/ruixliu/tmp/models/20200307_104151_0_bert_encoder_epoch_0001.pt

CUDA_VISIBLE_DEVICES=2,3 python run_finetuning_glue.py --model=bert --config_file="bert-base-single-node-rui.json" --config_file_path="/gpfs/gpfs0/groups/mozafari/ruixliu/code/distributed_learning/pytorch/src/bert/pretrain/configs" --data_dir=/gpfs/gpfs0/groups/mozafari/ruixliu/data/glue_data/RTE --task_name=rte --output_dir=/gpfs/gpfs0/groups/mozafari/ruixliu/tmp/output --do_train --do_eval --do_lower_case --checkpoint_file=/gpfs/gpfs0/groups/mozafari/ruixliu/tmp/models/20200307_104151_0_bert_encoder_epoch_0001.pt
