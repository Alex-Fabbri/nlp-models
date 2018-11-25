from typing import Dict
from overrides import overrides

#import numpy as np
import torch
import torch.nn.functional as F

from allennlp.training.metrics.metric import Metric

def perplexity_func(preds, targs):
    lprobs = F.log_softmax(preds, dim=-1)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    targs = targs.view(-1)
    loss_func = torch.nn.NLLLoss()
    loss = loss_func(lprobs, targs)
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
        # TODO check perplexity
        return {"perplexity": 0}
        #if len(self.losses) == 0:
        #    return {"perplexity": 0}
        #return {"perplexity": np.exp(np.mean(self.losses))}
