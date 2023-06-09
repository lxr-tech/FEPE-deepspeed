"""
  - BERT_base (L=12, H=768, A=12, #Para=110M) and BERT_large (L=24, H=1024, A=16, #Para=340M)
  - FNet hyper L=2, H=128, A=2; L=2, H=256, A=4; L=4, H=256, A=4; L=4, H=512, A= 8;
               L=8, H=256, A=4; L=8, H=512, A=8; L=12, H=512, A=8; L=12, H=768, A=12;
"""

configs = {
    ('sst2', 'transformer', False):  {
        'n_heads': 12, 'head_dims': 256, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'optim_type': 'sgd', 'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('sst2', 'fourier2d', True):  {
        'n_heads': 12, 'head_dims': 256, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'optim_type': 'sgd', 'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('sst2', 'fourier1d', True):  {
        'n_heads': 12, 'head_dims': 256, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'optim_type': 'sgd', 'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('stsb', 'transformer', False):  {
        'n_heads': 12, 'head_dims': 256, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'optim_type': 'sgd', 'lr': 0.0006, 'batch_size': 16, 'warmup_steps': 0.01,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
}
