set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 clm_train_en.py 'xpos_imp_1d_raw' 'full fepe -log'
