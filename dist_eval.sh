GPUS=4
accelerate launch --num_machines=1 --num_processes $GPUS --machine_rank 0 eval_sorce.py \
      --batch_size 1 \
      --llava_llama3 --use_e5v_rep --position_prompts --data './datasets/sorce-1k/dataset.jsonl' --name sorce-test