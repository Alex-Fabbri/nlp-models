from collections import Counter
import math
from typing import Iterable, Tuple, Dict, Set

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric
import torch.nn.functional as F
import numpy as np

def perplexity_func(preds, targs):
    lprobs =  F.log_softmax(preds, dim=-1)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    targs = targs.view(-1)
    l1 = torch.nn.NLLLoss()
    loss = l1(lprobs, targs)
    return loss

@Metric.register("perplexity")
class Perplexity(Metric):
    def __init__(self) -> None:
        self.losses = []

    @overrides
    def reset(self) -> None:
        self.losses = []

    @overrides
    def __call__(self,  # type: ignore
                 predictions: torch.LongTensor,
                 gold_targets: torch.LongTensor) -> None:
        cur_loss = perplexity_func(predictions, gold_targets)
        self.losses.append(cur_loss)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if reset:
            self.reset()
        if len(self.losses) == 0:
            return {"perplexity": 0}
        return {"perplexity": np.exp(np.mean(self.losses))}
