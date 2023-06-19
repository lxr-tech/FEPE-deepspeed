set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_train_en.py 'pre_rope_inv_2d_raw' 4 'rope original'
