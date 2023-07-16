set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:4,5,6,7 clm_train_pe.py 512 4 'xpos +imp' > clm_init_big_xpos_imp_2d_raw_1_512.log 2>&1
