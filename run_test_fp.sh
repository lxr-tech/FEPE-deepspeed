set -x
port=$(shuf -i25000-30000 -n1)

#WANDB_MODE=disabled \
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 clm_test_fp.py \
# --default 'fp32' --pe_fp 'fp32' --attn_fp 'bf16' --qk_fp 'bf16' --vo_fp 'bf16' --ffn_fp 'fp32' --norm_fp 'fp32' --key 'fp32_rope_inv_2d_raw_bf16_attn'
#wait
WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=4 clm_test_fp.py \
 --default 'fp32' --pe_fp 'fp32' --qk_fp 'fp16' --vo_fp 'fp32' --ffn_fp 'fp32' --norm_fp 'fp32' --key 'fp32_rope_inv_2d_raw_fp16_qk'
wait
WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=4 clm_test_fp.py \
 --default 'fp32' --pe_fp 'fp32' --qk_fp 'fp32' --vo_fp 'fp16' --ffn_fp 'fp32' --norm_fp 'fp32' --key 'fp32_rope_inv_2d_raw_fp16_vo'
wait
WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=4 clm_test_fp.py \
 --default 'fp32' --pe_fp 'fp32' --qk_fp 'fp16' --vo_fp 'fp16' --ffn_fp 'fp32' --norm_fp 'fp32' --key 'fp32_rope_inv_2d_raw_fp16_qkvo'
wait
WANDB_MODE=disabled \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=4 clm_test_fp.py \
 --default 'fp32' --pe_fp 'fp32' --qk_fp 'bf16' --vo_fp 'bf16' --ffn_fp 'fp32' --norm_fp 'fp32' --key 'fp32_rope_inv_2d_raw_bf16_qkvo'
#wait
#WANDB_MODE=disabled \
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 clm_test_fp.py \
# --default 'fp32' --pe_fp 'fp32' --attn_fp 'bf16' --qk_fp 'fp32' --vo_fp 'fp32' --ffn_fp 'fp32' --norm_fp 'fp32' --key 'fp32_rope_inv_2d_raw_bf16_attn'
#wait
#WANDB_MODE=disabled \
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 clm_test_fp.py \
# --default 'fp32' --pe_fp 'fp32' --attn_fp 'fp32' --ffn_fp 'fp32' --norm_fp 'bf16' --key 'fp32_rope_inv_2d_raw_bf16_norm'
#wait
#WANDB_MODE=disabled \
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 clm_test_fp.py \
# --default 'fp32' --pe_fp 'bf16' --attn_fp 'fp32' --ffn_fp 'fp32' --norm_fp 'fp32' --key 'fp32_rope_inv_2d_raw_bf16_pe'
#wait
#WANDB_MODE=disabled \
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 clm_test_fp.py \
# --default 'fp32' --pe_fp 'fp32' --attn_fp 'fp32' --ffn_fp 'bf16' --norm_fp 'fp32' --key 'fp32_rope_inv_2d_raw_bf16_ffn'
#wait
#WANDB_MODE=disabled \
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 clm_test_fp.py \
# --default 'fp32' --pe_fp 'fp32' --attn_fp 'fp32' --ffn_fp 'fp32' --norm_fp 'fp32' --key 'fp32_rope_inv_2d_raw_full'
