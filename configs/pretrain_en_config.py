"""
  - BERT_base (L=12, H=768, A=12, #Para=110M) and BERT_large (L=24, H=1024, A=16, #Para=340M)
  - FNet hyper L=2, H=128, A=2; L=2, H=256, A=4; L=4 , H=256, A=4; L=4 , H=512, A= 8;
               L=8, H=256, A=4; L=8, H=512, A=8; L=12, H=512, A=8; L=12, H=768, A=12;
  - batch size: 16, 32; Learning rate (Adam): 5e-5, 3e-5, 2e-5; Number of epochs: 2, 3, 4
"""

configs = {
    ('mlm_nsp', 'transformer', False):  {
        'num_layers': 4, 'd_model': 256, 'n_heads': 4, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.1,
        'batch_size': 32, 'lr': 0.0001, 'optim_type': 'adam', 'num_epoch': 100, 'weight_decay': 0.02,
        'warmup_steps': 0.01, 'mlm_probability': 0.15, 'nsp_probability': 0.5,
    },
}
