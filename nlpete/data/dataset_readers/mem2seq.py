import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("mem2seq")
class Mem2SeqDatasetReader(DatasetReader):

    def __init__(self,
                 target_namespace: str,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 truncate_source_len: int = None,
                 truncate_target_len: int = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._target_namespace = target_namespace
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self.truncate_source_len = truncate_source_len
        self.truncate_target_len = truncate_target_len

    @staticmethod
    def _read_line(line_num: int, line: str) -> Tuple[Optional[str], Optional[str]]:
        line = line.strip("\n")
        if not line:
            return None, None
        line_parts = line.split('\t')
        if len(line_parts) != 2:
            raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
        source_sequence, target_sequence = line_parts
        return source_sequence, target_sequence

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                source_sequence, target_sequence = self._read_line(line_num, line)
                if not source_sequence:
                    continue
                yield self.text_to_instance(source_sequence, target_sequence)

    @staticmethod
    def _create_target_to_source_sentinel_array(tokenized_source: List[Token],
                                                tokenized_target: List[Token]) -> np.array:
        target_to_source_sentinel_array: List[int] = []
        for target_token in tokenized_target[1:-1]:
            index = [loc for loc, val in enumerate(tokenized_source[1:-1]) if val == target_token]
            if index:
                index = max(index)
            else:
                index = len(tokenized_source[1:-1])
            target_to_source_sentinel_array.append([index])
        target_to_source_sentinel_array.append([len(tokenized_source)-1])
        return np.array(target_to_source_sentinel_array)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self.truncate_source_len is not None:
            tokenized_source = tokenized_source[:self.truncate_source_len]
        source_field = TextField(tokenized_source, self._source_token_indexers)
        fields_dict = {
                "source_tokens": source_field,
        }

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            if self.truncate_target_len is not None:
                tokenized_target = tokenized_target[:self.truncate_target_len+1]
            target_field = TextField(tokenized_target, self._target_token_indexers)
            target_to_source_sentinel_array = self._create_target_to_source_sentinel_array(\
                    tokenized_source, tokenized_target)
            target_to_source_sentinel_field = ArrayField(target_to_source_sentinel_array)


            fields_dict["target_tokens"] = target_field
            fields_dict["target_to_source_sentinel"] = target_to_source_sentinel_field

        return Instance(fields_dict)
