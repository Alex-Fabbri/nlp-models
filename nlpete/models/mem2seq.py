import logging
from typing import Dict, Tuple

from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.nn.beam_search import BeamSearch


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("mem2seq")
class Mem2Seq(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 copy_token: str = "@COPY@",
                 hidden_size: int = 100,
                 n_hops: int = 3,
                 dropout: float = 0.0,
                 beam_size: int = 5,
                 max_decoding_steps: int = 50,
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens") -> None:
        super(Mem2Seq, self).__init__(vocab)
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token, self._target_namespace)  # pylint: disable=protected-access
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
        self._copy_index = self.vocab.get_token_index(copy_token, self._target_namespace)
        if self._copy_index == self._oov_index:
            raise ConfigurationError(f"Special copy token {copy_token} missing from target vocab namespace. "
                                     f"You can ensure this token is added to the target namespace with the "
                                     f"vocabulary parameter 'tokens_to_add'.")

        # source_namespace and target_namespace are the same for my use case
        self._encoder = EncoderMemNN(self.vocab.get_vocab_size(self._source_namespace),
                                     hidden_size, n_hops, dropout, self._pad_index)
        self._decoder = DecoderMemNN(self.vocab.get_vocab_size(self._source_namespace),
                                     hidden_size, n_hops, dropout, self._pad_index)
        # TODO change to source when they are equal
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._source_namespace)
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None,
                target_to_source_sentinel: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        original_state = self._encode(source_tokens)
        # TODO make different depending on whether training vs validation
        # TODO add beam search
        # add sentinel copy token in preprocessing
        if target_tokens:
            state = self._init_decoder_state(original_state)
            output_dict = self._forward_loop(target_tokens, target_to_source_sentinel, state)
        else:
            output_dict = {}
        if not self.training:
            state = self._init_decoder_state(original_state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, _ = state["source_mask"].size()
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)
        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, state, self.take_search_step)
        return {
                "predicted_log_probs": log_probabilities,
                "predictions": all_top_k_predictions,
        }

    def take_search_step(self,
                         last_predictions: torch.Tensor,
                         state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        This function is what gets passed to the `BeamSearch.search` method. It takes
        predictions from the last timestep and the current state and outputs
        the log probabilities assigned to tokens for the next timestep, as well as the updated
        state.

        Since we are predicting tokens out of the extended vocab (target vocab + all unique
        tokens from the source sentence), this is a little more complicated that just
        making a forward pass through the model. The output log probs will have
        shape `(group_size, target_vocab_size + trimmed_source_length)` so that each
        token in the target vocab and source sentence are assigned a probability.

        Note that copy scores are assigned to each source token based on their position, not unique value.
        So if a token appears more than once in the source sentence, it will have more than one score.
        Further, if a source token is also part of the target vocab, its final score
        will be the sum of the generation and copy scores. Therefore, in order to
        get the score for all tokens in the extended vocab at this step,
        we have to combine copy scores for re-occuring source tokens and potentially
        add them to the generation scores for the matching token in the target vocab, if
        there is one.

        So we can break down the final log probs output as the concatenation of two
        matrices, A: `(group_size, target_vocab_size)`, and B: `(group_size, trimmed_source_length)`.
        Matrix A contains the sum of the generation score and copy scores (possibly 0)
        for each target token. Matrix B contains left-over copy scores for source tokens
        that do NOT appear in the target vocab, with zeros everywhere else. But since
        a source token may appear more than once in the source sentence, we also have to
        sum the scores for each appearance of each unique source token. So matrix B
        actually only has non-zero values at the first occurence of each source token
        that is not in the target vocab.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            Shape: `(group_size,)`

        state : ``Dict[str, torch.Tensor]``
            Contains all state tensors necessary to produce generation and copy scores
            for next step.

        Notes
        -----
        `group_size` != `batch_size`. In fact, `group_size` = `batch_size * beam_size`.
        """
        _, trimmed_source_length = state["source_to_target"].size()

        # Get input to the decoder RNN and the selective weights. `input_choices`
        # is the result of replacing target OOV tokens in `last_predictions` with the
        # copy symbol. `selective_weights` consist of the normalized copy probabilities
        # assigned to the source tokens that were copied. If no tokens were copied,
        # there will be all zeros.
        # shape: (group_size,), (group_size, trimmed_source_length)
        input_choices, selective_weights = self._get_input_and_selective_weights(last_predictions, state)
        # Update the decoder state by taking a step through the RNN.
        state = self._decoder_step(input_choices, selective_weights, state)
        # Get the un-normalized generation scores for each token in the target vocab.
        # shape: (group_size, target_vocab_size)
        generation_scores = self._get_generation_scores(state)
        # Get the un-normalized copy scores for each token in the source sentence,
        # excluding the start and end tokens.
        # shape: (group_size, trimmed_source_length)
        copy_scores = self._get_copy_scores(state)
        # Concat un-normalized generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # shape: (group_size, trimmed_source_length)
        copy_mask = state["source_mask"][:, 1:-1].float()
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat((generation_scores.new_full(generation_scores.size(), 1.0), copy_mask), dim=-1)
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        probs = util.masked_softmax(all_scores, mask)
        # shape: (group_size, target_vocab_size), (group_size, trimmed_source_length)
        generation_probs, copy_probs = probs.split([self._target_vocab_size, trimmed_source_length], dim=-1)
        # Update copy_probs needed for getting the `selective_weights` at the next timestep.
        state["copy_probs"] = copy_probs
        # We now have normalized generation and copy scores, but to produce the final
        # score for each token in the extended vocab, we have to go through and add
        # the copy scores to the generation scores of matching target tokens, and sum
        # the copy scores of duplicate source tokens.
        # shape: (group_size, target_vocab_size + trimmed_source_length)
        final_probs = self._gather_final_probs(generation_probs, copy_probs, state)

        return final_probs.log(), state
    def _encode(self,
                source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode source input sentences.
        """
        encoder_outputs = self._encoder(source_tokens["tokens"]).unsqueeze(0)
        source_mask = util.get_text_field_mask(source_tokens)
        return {
                "source_mask": source_mask,
                "source_tokens": source_tokens["tokens"],
                "encoder_outputs": encoder_outputs
                }

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        story = []
        for hop in range(self._decoder.max_hops):
            embed_a = self._decoder.c_list[hop](state["source_tokens"].contiguous().view(\
                        state["source_tokens"].size(0), -1))#.long()) # b * (m * s) * e
            m_a = embed_a
            embed_c = self._decoder.c_list[hop+1](state["source_tokens"].contiguous().view(\
                    state["source_tokens"].size(0), -1).long())
            m_c = embed_c
            story.append(m_a)
        story.append(m_c)
        story = torch.stack(story)

        state["story"] = story
        state["decoder_hidden"] = state["encoder_outputs"]
        state["decoder_context"] = state["encoder_outputs"].new_zeros(state["encoder_outputs"].size())
        return state

    def _forward_loop(self,
                      target_tokens: Dict[str, torch.LongTensor],
                      target_to_source_sentinel: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss against gold targets.
        """
        _, target_sequence_length = target_tokens["tokens"].size()
        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1
        all_decoder_outputs_ptr = []
        all_decoder_outputs_vocab = []
        # TODO add non-teacher forcing
        for timestep in range(num_decoding_steps):
            input_choices = target_tokens["tokens"][:, timestep]
            decoder_ptr, decoder_vocab, decoder_hidden = \
                    self._decoder(input_choices, state)
            state["decoder_hidden"] = decoder_hidden
            all_decoder_outputs_ptr.append(decoder_ptr)
            all_decoder_outputs_vocab.append(decoder_vocab)
        all_decoder_outputs_ptr_tensor = torch.stack(all_decoder_outputs_ptr)
        all_decoder_outputs_vocab_tensor = torch.stack(all_decoder_outputs_vocab)

        target_mask = util.get_text_field_mask(target_tokens)
        target_lengths = util.get_lengths_from_binary_sequence_mask(target_mask)
        # account for SOS token
        target_lengths = target_lengths - 1
        loss_vocab = masked_cross_entropy(
                all_decoder_outputs_vocab_tensor.transpose(0, 1).contiguous(), # -> batch x seq
                target_tokens["tokens"][:, 1:].contiguous(), # -> batch x seq
                target_lengths
        )
        loss_ptr = masked_cross_entropy(
                all_decoder_outputs_ptr_tensor.transpose(0, 1).contiguous(), # -> batch x seq
                target_to_source_sentinel.squeeze().contiguous().long(), # -> batch x seq
                target_lengths
        )

        loss = loss_vocab + loss_ptr
        return {"loss": loss}

class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, pad_index):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.c_list = nn.ModuleList()
        for _ in range(self.max_hops+1):
            c_cur = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=pad_index)
            c_cur.weight.data.normal_(0, 0.1)
            self.c_list.append(c_cur)
        self.softmax = nn.Softmax(dim=1)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if torch.cuda.is_available():
            return torch.zeros(bsz, self.embedding_dim).cuda()
        else:
            return torch.zeros(bsz, self.embedding_dim)

#pylint: disable=arguments-differ
    def forward(self, story):
        # u  = torch.Size([32, 100])
        u_list = [self.get_state(story.size(0))]
        for hop in range(self.max_hops):
            # torch.Size([32, 19, 100])
            embed_a = self.c_list[hop](story.contiguous().view(story.size(0), -1).long()) # b * (m * s) * e
            #embed_a = self.c_list[hop](story.view(story.size(0), -1)) # b * (m * s) * e
            u_temp = u_list[-1].unsqueeze(1).expand_as(embed_a)
            prob = self.softmax(torch.sum(embed_a*u_temp, 2))
            embed_c = self.c_list[hop+1](story.contiguous().view(story.size(0), -1).long())
            #embed_c = self.c_list[hop+1](story.view(story.size(0), -1))
            prob = prob.unsqueeze(2).expand_as(embed_c)
            # o_k = torch.Size([32, 100])
            o_k = torch.sum(embed_c*prob, 1)
            # u_k = 32 x 100, just return the last one
            u_k = u_list[-1] + o_k
            u_list.append(u_k)
        return u_k

#pylint: disable=abstract-method
class DecoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, pad_index):
        super(DecoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.c_list = nn.ModuleList()
        for _ in range(self.max_hops+1):
            cur_c = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=pad_index)
            cur_c.weight.data.normal_(0, 0.1)
            self.c_list.append(cur_c)
        self.softmax = nn.Softmax(dim=1)
        self.w_1 = nn.Linear(2*embedding_dim, self.num_vocab)
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)

    #pylint: disable=arguments-differ
    def forward(self, enc_query, state):
        # enc_query = size [32]
        # embed_q = torch.Size([32, 100])
        embed_q = self.c_list[0](enc_query) # b * e
        _, hidden = self.gru(embed_q.unsqueeze(0), state["decoder_hidden"])
        temp = []
        u_list = [hidden[0].squeeze()]
        for hop in range(self.max_hops):
            m_a = state["story"][hop]
            u_temp = u_list[-1].unsqueeze(1).expand_as(m_a)
            # prob_lg = size 32x16 (eg)
            prob_lg = torch.sum(m_a*u_temp, 2)
            prob_ = self.softmax(prob_lg)
            m_c = state["story"][hop+1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_c)
            o_k = torch.sum(m_c*prob, 1)
            if hop == 0:
                p_vocab = self.w_1(torch.cat((u_list[0], o_k), 1))
            u_k = u_list[-1] + o_k
            u_list.append(u_k)
        p_ptr = prob_lg
        return p_ptr, p_vocab, hidden

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).cuda().long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns
    -------
        loss: An average loss value masked by the length.
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.contiguous().view(-1, logits.size(-1)) ## -1 means infered from other dimentions
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.contiguous().view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.contiguous().view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss
