from typing import Dict, List, Tuple, Union, Optional
from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, remove_sentence_boundaries
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.training.metrics import Perplexity, Metric

from fairseq.models.transformer import TransformerDecoder, TransformerLanguageModel, Embedding
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from fairseq import options

# tools for wrapping fairseq model
class FairseqDictionary():
    def __init__(self, vocab):
        self.vocab = vocab
    def __len__(self):
        return self.vocab.get_vocab_size()

class AttributeDict(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

@Model.register('fairseq_transformer_lm')
class FairseqTransformerLM(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 decoder_embed_dim: int, 
                 decoder_output_dim: int,
                 decoder_input_dim: int,
                 decoder_ffn_embed_dim: int,
                 decoder_layers: int,
                 decoder_attention_heads: int,
                 eval_metric: Metric=None,
                 dropout: float=0.1,
                 attention_dropout: float=0.,
                 relu_dropout: float=0.,
                 adaptive_softmax_cutoff: bool=None, 
                 adaptive_softmax_dropout: float=None,
                 adaptive_softmax_factor: float=None,
                 no_token_positional_embeddings: bool=False,
                 share_decoder_input_output_embed: bool=False, 
                 character_embeddings: bool=False, 
                 character_filters: str=None,
                 character_embed_dim: int=4,
                 char_embedder_highway_layers: int=2,
                 adaptive_input: bool=None,
                 adaptive_input_factor: float=None,
                 adaptive_input_cutoff: str=None, 
                 tie_adaptive_weights: bool=None, 
                 tie_adaptive_proj: bool=None, 
                 decoder_learned_pos: bool=False, 
                 decoder_normalize_before: bool=True,
                 no_tie_adaptive_proj: bool=None,
                 tokens_per_sample: int=64) -> None:
        super().__init__(vocab)
        # TODO: things to add to match fairseq:
        # 1. add lr certain lr schedulers (check if reduce_on_plateau is here)
        # 2. lr_shrink=0.1
        # 3. max_tokens = 6000
        # 4.  add clip_norm = 25
        # 5. sentence_avg=False
        # 6  min_loss_scale=0.0001, min_lr=1e-05, momentum=0.99
        # 7. criterion = cross_entropy
        # 8. change tokens_per_sample -- 1024 in fairseq code
        # 9. change so adaptive softmax and input work
        self.vocab = vocab
        self.decoder_input_dim = decoder_input_dim
        self.eval_metric = eval_metric
        args = AttributeDict(vars())
        if no_tie_adaptive_proj in args and no_tie_adaptive_proj == False:
            # backward compatibility
            args.tie_adaptive_proj = True
        if tokens_per_sample is not None:
            args.max_source_positions = tokens_per_sample
            args.max_target_positions = tokens_per_sample

        fairseq_dict = FairseqDictionary(vocab)
        # TODO refactor embedders into allennlp
        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(fairseq_dict, eval(args.character_filters),
                                                  args.character_embedding_dim,
                                                  args.decoder_embed_dim,
                                                  args.char_embedder_highway_layers,
                                                  )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(len(fairseq_dict), vocab.get_token_index(DEFAULT_PADDING_TOKEN), \
                            args.decoder_input_dim, args.adaptive_input_factor, args.decoder_embed_dim, options.eval_str_list(args.adaptive_input_cutoff, type=int))
        else:
            embed_tokens = Embedding(vocab.get_vocab_size(), self.decoder_input_dim, vocab.get_token_index(DEFAULT_PADDING_TOKEN))

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.decoder_input_dim == args.decoder_output_dim
        self.decoder = TransformerDecoder(args,  fairseq_dict, embed_tokens, no_encoder_attn=True, final_norm=False)


    def forward(self,  # type: ignore
                input_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        By convention, the input dict is required to have at least a ``"tokens"``
        entry that's the output of a ``SingleIdTokenIndexer``, which is used
        to compute the language model targets.

        If the model was instantatiated with ``remove_bos_eos=True``,
        then it is expected that each of the input sentences was augmented with
        begin-sentence and end-sentence tokens.

        """
        # We must have token_ids so that we can compute targets
        token_ids = input_tokens.get("tokens")
        if token_ids is None:
            raise ConfigurationError("Your data must have a 'tokens': SingleIdTokenIndexer() "
                                     "in order to use the BidirectionalLM")

        # Use token_ids to compute targets
        forward_targets = torch.zeros_like(token_ids)
        forward_targets[:, 0:-1] = token_ids[:, 1:]

        output = self.decoder(token_ids)
        output_loss = output[0]

        # calculate perplexity
        if not self.training: 
            self.eval_metric(output_loss.detach(), forward_targets.detach())

        lprobs =  F.log_softmax(output_loss, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        forward_targets = forward_targets.view(-1)
        l1 = torch.nn.NLLLoss()
        loss = l1(lprobs, forward_targets)
        return_dict = {'loss': loss}


        return return_dict
    # https://github.com/epwalsh/nlp-models/blob/master/nlpete/training/metrics/bleu.py
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update(self.eval_metric.get_metric(reset=reset))
        return all_metrics
