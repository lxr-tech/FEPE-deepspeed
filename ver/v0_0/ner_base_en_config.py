"""
  - BERT_base (L=12, H=768, A=12, #Para=110M) and BERT_large (L=24, H=1024, A=16, #Para=340M)
  - FNet hyper L=2, H=128, A=2; L=2, H=256, A=4; L=4, H=256, A=4; L=4, H=512, A= 8;
               L=8, H=256, A=4; L=8, H=512, A=8; L=12, H=512, A=8; L=12, H=768, A=12;
"""

configs = {
    ('conll2003', 'transformer', False):  {  # fixed
        'n_heads': 12, 'head_dims': 128, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0001, 'batch_size': 32, 'warmup_steps': 0.01, 'optim_type': 'adam',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.4, 'fc_dropout': 0.4,
    },
    ('conll2003', 'adatrans', False): {  # fixed
        'n_heads': 12, 'head_dims': 128, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 2,
        'lr': 0.0001, 'batch_size': 32, 'warmup_steps': 0.01, 'optim_type': 'adam',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.4, 'fc_dropout': 0.4,
    },
    ('conll2003', 'rotary', False): {  # need tuning
        'n_heads': 12, 'head_dims': 128, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0015, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd', 'dropout': 0.15,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'fc_dropout': 0.4,
    },
    ('conll2003', 'fourier2d', False): {
        'n_heads': 12, 'head_dims': 128, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0001, 'batch_size': 32, 'warmup_steps': 0.01, 'optim_type': 'adam',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4,
    },
    ('conll2003', 'exponential', False): {
        'n_heads': 12, 'head_dims': 128, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0015, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd', 'dropout': 0.15,
        # 'lr': 0.00005, 'batch_size': 32, 'warmup_steps': 0.01, 'optim_type': 'adam', 'dropout': 0.4,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'fc_dropout': 0.4,
    },
    ('conll2003', 'exp_flash', False): {
        'n_heads': 24, 'head_dims': 64, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        # 'lr': 0.0015, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd', 'dropout': 0.15,
        'lr': 0.0001, 'batch_size': 32, 'warmup_steps': 0.01, 'optim_type': 'adam', 'dropout': 0.4,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'fc_dropout': 0.4,
    },
    ('conll2003', 'exp_performer', False): {
        'n_heads': 12, 'head_dims': 128, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0015, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd', 'dropout': 0.15,
        # 'lr': 0.000015, 'batch_size': 32, 'warmup_steps': 0.01, 'optim_type': 'adam', 'dropout': 0.4,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'fc_dropout': 0.4,
    },
    ('en-ontonotes', 'transformer', False): {  # fixed
        'n_heads': 10, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4,
    },
    ('en-ontonotes', 'adatrans', False): {  # fixed
        'n_heads': 10, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 2,
        'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4,
    },
    ('en-ontonotes', 'rotary', False): {  #
        'n_heads': 10, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4,
    },
    ('en-ontonotes', 'fourier2d', False): {
        'n_heads': 10, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4,
    },
    ('en-ontonotes', 'exponential', False): {
        'n_heads': 10, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.002, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd', 'dropout': 0.15,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'fc_dropout': 0.4,
    },
    ('en-ontonotes', 'exp_flash', False): {
        'n_heads': 15, 'head_dims': 64, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd', 'dropout': 0.15,
        # 'lr': 0.00005, 'batch_size': 32, 'warmup_steps': 0.01, 'optim_type': 'adam', 'dropout': 0.4,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'fc_dropout': 0.4,
    },
    ('en-ontonotes', 'exp_performer', False): {
        'n_heads': 10, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.002, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd', 'dropout': 0.15,
        # 'lr': 0.0001, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'adam', 'dropout': 0.15,
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'fc_dropout': 0.4,
    },
}

'''configs = {
    ('conll2003', 'transformer', False):  {  # fixed
        'n_heads': 12, 'head_dims': 256, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('conll2003', 'adatrans', False): {  # fixed
        'n_heads': 14, 'head_dims': 128, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 2,
        'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',  # conll2003的lr不能超过0.002
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('conll2003', 'rotary', False): {  #
        'n_heads': 14, 'head_dims': 128, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'lr': 0.0015, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('conll2003', 'rotary_v', False): {  #
        'n_heads': 14, 'head_dims': 128, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'lr': 0.0018, 'batch_size': 16, 'warmup_steps': 0.1, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('conll2003', 'fourier2d', False): {
        'n_heads': 12, 'head_dims': 256, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'lr': 0.0009, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('conll2003', 'fourier2d', True): {
        'n_heads': 12, 'head_dims': 256, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'lr': 0.0006, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('conll2003', 'fourier1d', False): {
        'n_heads': 14, 'head_dims': 128, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'lr': 0.0015, 'batch_size': 16, 'warmup_steps': 0.1, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('conll2003', 'fourier1d', True): {
        'n_heads': 14, 'head_dims': 128, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'lr': 0.0006, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('conll2003', 'exponential', False): {
        'n_heads': 14, 'head_dims': 128, 'num_layers': 2, 'char_type': 'cnn', 'ffn_dim_rate': 4,
        'lr': 0.001, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('en-ontonotes', 'transformer', False): {  # fixed
        'n_heads': 8, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0007, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('en-ontonotes', 'adatrans', False): {  # fixed
        'n_heads': 8, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 2,
        'lr': 0.0007, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('en-ontonotes', 'rotary', False): {  #
        'n_heads': 8, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.001, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('en-ontonotes', 'fourier2d', False): {
        'n_heads': 8, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0007, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('en-ontonotes', 'fourier2d', True): {
        'n_heads': 8, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0007, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('en-ontonotes', 'fourier1d', False): {
        'n_heads': 8, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0007, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('en-ontonotes', 'fourier1d', True): {
        'n_heads': 8, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.0004, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
    ('en-ontonotes', 'exponential', False): {
        'n_heads': 8, 'head_dims': 96, 'num_layers': 2, 'char_type': 'adatrans', 'ffn_dim_rate': 4,
        'lr': 0.001, 'batch_size': 16, 'warmup_steps': 0.01, 'optim_type': 'sgd',
        'model_type': 'transformer', 'normalize_embed': True, 'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.4
    },
}
'''


class Config:
    def __init__(self, **kwargs):
        for item in kwargs:
            self.__setattr__(item, kwargs[item])


if __name__ == '__main__':
    configs = Config(**configs['conll2003'])
    print(configs.__dir__())
