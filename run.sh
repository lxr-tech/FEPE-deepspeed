set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_train_pe.py 'init_pre_rope_inv_2d_raw_fp32' 4 'rope in fp32' > clm_init_pre_rope_inv_2d_raw_1_512_fp32.log 2>&1
