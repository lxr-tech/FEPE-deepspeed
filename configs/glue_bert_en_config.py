"""
  - BERT_base (L=12, H=768, A=12, #Para=110M) and BERT_large (L=24, H=1024, A=16, #Para=340M)
  - FNet hyper L=2, H=128, A=2; L=2, H=256, A=4; L=4, H=256, A=4; L=4, H=512, A= 8;
               L=8, H=256, A=4; L=8, H=512, A=8; L=12, H=512, A=8; L=12, H=768, A=12;
  - batch size: 16, 32; Learning rate (Adam): 5e-5, 3e-5, 2e-5; Number of epochs: 2, 3, 4
"""

configs = {
    ('cola', 'bert-base-cased', False):  {  # done
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
    ('sst2', 'bert-base-cased', False):  {  # done, temporarily
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
    ('sst2', 'junnyu/roformer_chinese_base', False):  {  #
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
    ('mrpc', 'bert-base-cased', False):  {  # done
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 32,
    },
    ('mrpc', 'junnyu/roformer_chinese_base', False):  {  #
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
    ('stsb', 'bert-base-cased', False):  {  # done
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
    ('stsb', 'junnyu/roformer_chinese_base', False):  {  #
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
    ('qqp', 'bert-base-cased', False):  {
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
    ('mnli_m', 'bert-base-cased', False):  {
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
    ('mnli_mm', 'bert-base-cased', False):  {
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
    ('qnli', 'bert-base-cased', False):  {
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
    ('rte', 'bert-base-cased', False):  {  # done
        'optim_type': 'adam', 'lr': 2e-5, 'batch_size': 16,
    },
}
