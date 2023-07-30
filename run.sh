set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_train_pe.py 'new_init_pre_rope_imp_1d_raw' 4 'rope+imp+1d with new range' > clm_new_init_pre_rope_imp_1d_raw_1_512.log 2>&1
wait
WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 clm_train_pe.py 'new_init_pre_xpos_imp_1d_raw' 4 'xpos+imp+1d with new range' > clm_new_init_pre_xpos_imp_1d_raw_1_512.log 2>&1
