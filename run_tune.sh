set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_tune_pe.py 'init_pre_rope_inv_2d_raw' > clm_lora_init_pre_rope_inv_2d_raw_1_512.log 2>&1
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:4,5,6,7 clm_tune_pe.py 'init_pre_rope_imp_2d_raw' > clm_lora_init_pre_rope_imp_2d_raw_1_512.log 2>&1
