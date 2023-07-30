set -x
port=$(shuf -i25000-30000 -n1)

WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 clm_train_fp.py \
 --default 'fp32' --pe_fp 'fp32' --qk_fp 'bf16' --vo_fp 'fp32' --ffn_fp 'fp32' --norm_fp 'fp32' \
 --key 'fp32_rope_inv_2d_raw_bf16_qk' --modification 'post_init pre_norm test fp vs le' > clm_fp32_qk_bf16_rope_inv_2d_raw_1_512.log 2>&1
#wait
#WANDB_MODE=disabled \
#deepspeed --master_port "$port" --include localhost:4,5,6,7 clm_train_fp.py \
# --default 'fp32' --pe_fp 'fp32' --attn_fp 'fp32' --ffn_fp 'bf16' --norm_fp 'fp32' \
# --key 'fp32_rope_inv_2d_raw_bf16_ffn' --modification 'post_init pre_norm test fp vs le' > clm_fp32_ffn_bf16_rope_inv_2d_raw_1_512.log 2>&1
