import os
import shutil


checkpoints = {
    # 'rope_inv_2d_raw': ('ds_clm_rope_arxiv_1_512', 'checkpoints-230531', 'checkpoint-16920'),  # 8460
    'rope_inv_2d_log': ('ds_clm_rope_arxiv_1l_512', 'checkpoints-230531', 'checkpoint-16920'),  # 12690
    'rope_inv_1d_raw': ('ds_clm_rope_arxiv_1d_512', 'checkpoints-230531', 'checkpoint-16920'),  # 8460
    'rope_inv_1d_log': ('ds_clm_rope_arxiv_1dl_512', 'checkpoints-230531', 'checkpoint-16920'),  # (21150)

    # 'xpos_inv_2d_raw': ('ds_clm_fepe_arxiv_1s_512', 'checkpoints-230609', 'checkpoint-16920'),  # 12690
    'xpos_inv_2d_log': ('ds_clm_xpos_arxiv_1l_512', 'checkpoints-230531', 'checkpoint-16920'),  # 16920
    'xpos_inv_1d_raw': ('ds_clm_xpos_arxiv_1d_512', 'checkpoints-230531', 'checkpoint-16920'),  # 16920
    'xpos_inv_1d_log': ('ds_clm_xpos_arxiv_1dl_512', 'checkpoints-230531', 'checkpoint-12690'),  # 16920

    'rope_imp_2d_raw': ('ds_clm_fepe_arxiv_1q_512', 'checkpoints-230609', 'checkpoint-16920'),  # 8460
    'rope_imp_2d_log': ('ds_clm_fepe_arxiv_1t_512', 'checkpoints-230609', 'checkpoint-16920'),
    'rope_imp_1d_raw': ('ds_clm_fepe_arxiv_1n_512', 'checkpoints-230609', 'checkpoint-16920'),  # 12690
    'rope_imp_1d_log': ('ds_clm_fepe_arxiv_1m_512', 'checkpoints-230609', 'checkpoint-16920'),  # (21150)

    'xpos_imp_2d_raw': ('ds_clm_fepe_arxiv_1o_512', 'checkpoints-230609', 'checkpoint-16920'),
    'xpos_imp_2d_log': ('ds_clm_fepe_arxiv_1p_512', 'checkpoints-230609', 'checkpoint-16920'),  # (21150)
    'xpos_imp_1d_raw': ('ds_clm_fepe_arxiv_1j_512', 'checkpoints-230609', 'checkpoint-12690'),
    'xpos_imp_1d_log': ('ds_clm_fepe_arxiv_1l_512', 'checkpoints-230609', 'checkpoint-12690'),
}

names = ['checkpoint-4230', 'checkpoint-8460', 'checkpoint-12690', 'checkpoint-16920', 'checkpoint-21150']

for key in checkpoints:
    log, dct, _ = checkpoints[key]
    dst = f'/remote-home/xrliu/projects/FEPE-deepspeed/saved/230609/{key}/'
    os.mkdir(dst)
    for i, epoch in enumerate(names):
        src = f'/remote-home/xrliu/projects/FEPE-deepspeed/{dct}/{log}/{epoch}/pytorch_model.bin'
        old = f'/remote-home/xrliu/projects/FEPE-deepspeed/saved/230609/{key}/pytorch_model.bin'
        new = f'/remote-home/xrliu/projects/FEPE-deepspeed/saved/230609/{key}/pytorch_model_epoch{i+1}.bin'
        shutil.move(src, dst)
        os.rename(old, new)
    dst = f'/remote-home/xrliu/projects/FEPE-deepspeed/saved/230609/'
    old = f'/remote-home/xrliu/projects/FEPE-deepspeed/saved/230609/{log}.log'
    new = f'/remote-home/xrliu/projects/FEPE-deepspeed/saved/230609/{key}.log'
    log = f'/remote-home/xrliu/projects/FEPE-deepspeed/{log}.log'
    shutil.move(log, dst)
    os.rename(old, new)




