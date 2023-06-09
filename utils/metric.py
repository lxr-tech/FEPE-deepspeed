import torch
import torch.nn.functional as f

from transformers.trainer_pt_utils import nested_numpify, nested_concat

# todo:
#  https://discuss.huggingface.co/t/how-to-define-the-compute-metrics-function-in-trainer/12953


def compute_metrics_for_clm(eval_pred):
    logits, labels = eval_pred
    logits = torch.from_numpy(logits)[..., :-1, :]
    target = torch.from_numpy(labels)[..., 1:]
    pred = torch.max(logits, dim=-1)[1]

    mask = target.eq(0).float()
    batch_size, seq_len, vocab_size = logits.shape

    loss = f.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1),
                           ignore_index=0, reduction='none').resize_as(target)
    weight = mask / torch.sum(mask, dim=-1, keepdim=True)

    rank = torch.distributed.get_rank()

    if rank == 0:
        print(loss.shape, weight.shape)

    loss = torch.sum(loss * weight, dim=-1)

    acc = nested_numpify(torch.sum(pred.eq(labels).float() * mask) / torch.sum(mask))
    ppl = nested_numpify(torch.mean(torch.exp(loss)))

    if rank == 0:
        print(acc.shape, ppl.shape)

    return {'acc': acc, 'ppl': ppl}
