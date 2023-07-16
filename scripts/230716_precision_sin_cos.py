import torch
import scipy
import numpy as np


def count_idea(max_len, imag=False):
    t = np.arange(max_len).reshape((-1, ))
    si1, ci1 = scipy.special.sici(t)
    si2, ci2 = scipy.special.sici(t / 10000)
    if imag:
        idea = 1 / np.log(10000) * (si1 - si2)  # ci1 can be neglected
    else:
        idea = 1 / np.log(10000) * (ci1 - ci2)  # ci1 can be neglected
    return idea


def count_rope(max_len, head_dim, scaled, imag=False, dtype=torch.float32, device='cuda:0'):
    t = torch.arange(max_len, dtype=dtype, device=device).reshape((-1, 1))
    i = torch.linspace(1, head_dim // 2, head_dim // 2, dtype=dtype, device=device).reshape((1, -1))
    if imag:
        rope = torch.sin(10000 ** (-2 * i / head_dim) * t).to(dtype=dtype, device=device)
    else:
        rope = torch.cos(10000 ** (-2 * i / head_dim) * t).to(dtype=dtype, device=device)
    rope = rope * ((2 * i + 0.4 * head_dim) / (1.4 * head_dim)) ** (t / 512) if scaled else rope
    return torch.mean(rope, dim=-1).to(dtype=dtype, device=device)


head_dim = 64
max_len = 2048

idea = count_idea(max_len)
fp32 = count_rope(max_len, head_dim, scaled=False, imag=False, dtype=torch.float32)
fp16 = count_rope(max_len, head_dim, scaled=False, imag=False, dtype=torch.float16)
bf16 = count_rope(max_len, head_dim, scaled=False, imag=False, dtype=torch.bfloat16)

max_lens = [128, 512, 1024, 2048]

for max_len in max_lens:
    print(f'idea : Δt = {max_len} ,', idea[max_len-1])
    print(f'fp32 : Δt = {max_len} ,', fp32[max_len-1])
    print(f'fp16 : Δt = {max_len} ,', fp16[max_len-1])
    print(f'bf16 : Δt = {max_len} ,', bf16[max_len-1])
