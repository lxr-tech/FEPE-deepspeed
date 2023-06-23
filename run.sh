set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:4,5,6,7 clm_train_en.py 'pre_rope_inv_2d_log' 4 'rope +log' > clm_pre_rope_inv_2d_log_1_512.log 2>&1
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:4,5,6,7 clm_train_en.py 'pre_rope_inv_1d_log' 4 'rope +1d+log' > clm_pre_rope_inv_1d_log_1_512.log 2>&1
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:4,5,6,7 clm_train_en.py 'pre_xpos_inv_2d_log' 4 'xpos +log' > clm_pre_xpos_inv_2d_log_1_512.log 2>&1
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:4,5,6,7 clm_train_en.py 'pre_xpos_inv_1d_log' 4 'xpos +1d+log' > clm_pre_xpos_inv_1d_log_1_512.log 2>&1
