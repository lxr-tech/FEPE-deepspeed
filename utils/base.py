import torch
import numpy as np
import random


# def setup_seed(seed=42):  # https://blog.csdn.net/weixin_43135178/article/details/118768531
#     torch.manual_seed(seed)  # CPU random seed
#     np.random.seed(seed)  # numpy random seed
#     random.seed(seed)  # python random module.
#     if torch.cuda.is_available():
#         # torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#         torch.cuda.manual_seed(seed)  # gpu random seed for current
#         torch.cuda.manual_seed_all(seed)  # gpu random seed for all available


def setup_seed(seed: int = None, random: bool = True, numpy: bool = True,
               pytorch: bool = True, deterministic: bool = True):
    if seed is None:
        import time
        seed = int(time.time() % 1000000)
        print(seed)
    if random:
        import random
        random.seed(seed)
    if numpy:
        try:
            import numpy
            numpy.random.seed(seed)
        except:
            pass
    if pytorch:
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
        except:
            pass
    return seed
