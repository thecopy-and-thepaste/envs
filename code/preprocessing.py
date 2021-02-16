#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Remember to pdate the PYTHON_PATH to
# export PYTHONPATH=`pwd`:`pwd`/conabio_ml_text/conabio_ml:`pwd`/conabio_ml_text
from conabio_ml_text.conabio_ml_text.preprocessing.preprocessing import Tokens
import re

import timeit

from collections import defaultdict
from typing import Iterable, List

from conabio_ml.datasets.dataset import Dataset
from conabio_ml_text.preprocessing import BasePrePreprocessing
from conabio_ml_text.preprocessing.preprocessing import NUM_PROCESSES
from conabio_ml_text.utils.utils import poolify
from conabio_ml.utils.utils import Chained

from conabio_ml.utils.logger import get_logger, debugger

debug = debugger.debug


class BPE(BasePrePreprocessing):

    # region BPE implementation
    @staticmethod
    def process_words(dataset_words: str) -> defaultdict:
        vocab = defaultdict(int)
        ds = dataset_words.split()

        for word in ds:
            vocab[" ".join(list(word)) + " </w>"] += 1

        return vocab

    @staticmethod
    def get_stats(vocab):
        pairs = defaultdict(int)

        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    @staticmethod
    def merge_vocab(pair, v_in):
        v_out = {}
        bigram = re.escape(" ".join(pair))

        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    @staticmethod
    def get_tokens(vocab):
        tokens = defaultdict(int)

        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                tokens[token] += freq
        return tokens

    @staticmethod
    def get_tokens(vocab):
        tokens = defaultdict(int)

        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                tokens[token] += freq
        return tokens

    @staticmethod
    def update_vocab(bpe_vocab: dict,
                     subword_tokens: dict,
                     special_tokens: List[str] = []):
        vocab = defaultdict(int)
        len_sp_tokens = len(special_tokens)

        if len_sp_tokens > 0:
            vocab.update(dict([(special_tokens[ix], ix)
                               for ix in range(len_sp_tokens)]))

        for token in bpe_vocab:
            if token not in vocab:
                token = token.replace(" ", "")
                vocab[token] = len(vocab)

        for token in subword_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        return vocab

    @staticmethod
    def bpe(all_words: str,
            num_merges: int = 10,
            vocab_size: int = 0):
        # BPR consists on
        # 1. Further preproc of word to "word" -> "w o r d </w>"
        # 2. create vocab, subwords
        # 3. create inverse dictionaty
        all_vocab = BPE.process_words(all_words)

        vocab = all_vocab
        if vocab_size > 0:
            vocab = {k: v for k, v
                     in sorted(vocab.items(), key=lambda x: x[1], reverse=True)[0:vocab_size]}

        # Every merge creates a new pair of 2-size chars to merge with the vocab
        for i in range(num_merges):
            pairs = BPE.get_stats(all_vocab)
            # We get the most frequent
            best = max(pairs, key=pairs.get)

            # And update the vocab
            all_vocab = BPE.merge_vocab(best, all_vocab)

        # Finally, we obtain the tokens of subwords
        tokens = BPE.get_tokens(all_vocab)

        return tokens, vocab

    # end region
    @staticmethod
    def tokenize_word(word: str,
                      tokens: dict,
                      unk_token: str = Tokens.UNK_TOKEN):
        try:
            res_token = []
            matches = False

            if len(word) == 0:
                return []

            for k in tokens:
                matches = re.search(k, word)
                if matches:
                    match_token = matches.group(0)
                    init = BPE.tokenize_word(word[0:matches.start()], tokens)
                    end = BPE.tokenize_word(word[matches.end():], tokens)
                    res_token = init + [match_token] + end
                    break

            if not matches:
                res_token = [unk_token]

            return res_token
        except Exception as ex:
            print(ex)

    @staticmethod
    def tokenize_document(tokenize_args: dict = {
            "vocab": {},
            "tokens": {},
            "unk_token": Tokens.UNK_TOKEN},
            document: str = ""):
        try:
            all_tokens = [x+"</w>" for x in document.split()]
            vocab = tokenize_args.get("vocab", {})
            tokens = tokenize_args.get("tokens", {})
            unk_token = tokenize_args.get("unk_token", Tokens.UNK_TOKEN)

            converted_tokens = []

            assert len(vocab) > 0,\
                "You need to specify the vocab to tokenize the document"
            assert len(vocab) > 0,\
                "You need to specify the subword tokens to tokenize the document"

            for token in all_tokens:
                token = token.replace(" ", "")
                if token in vocab:
                    converted_tokens.append(token)
                else:
                    # If the token is not in the vocab we create a subword representation
                    converted_tokens += BPE.tokenize_word(token, tokens)

            return " ".join(converted_tokens)
        except Exception as ex:
            print(ex)

    @staticmethod
    def tokenize(data: Iterable,
                 vocab: dict,
                 tokens: dict,
                 unk_token: str = Tokens.UNK_TOKEN):
        res = poolify(data,
                      func_args={
                          "vocab": vocab,
                          "tokens": tokens,
                          "unk_token": unk_token},
                      fn=BPE.tokenize_document)

        return res
    # region bpe tokenization

    # endregion
    @staticmethod
    def preprocess_document(preprocess_args: dict,
                            data: str):
        vocab = defaultdict(int)
        ds = data.split()

        for word in ds:
            vocab[" ".join(list(word)) + " </w>"] += 1

        return vocab

    @staticmethod
    def preprocess(dataset,
                   preprocess_args: dict = {
                       "field": "item",
                       "num_merges": 10
                   },
                   vocab_args: dict = {
                       "size": 100,
                   },
                   preproc_args: dict = {}):
        field = preprocess_args.get("field", "item")
        num_merges = preprocess_args.get("num_merges", 10)
        vocab_size = vocab_args.get("size", 1000)

        data = dataset.data
        data = data[field]

        all_words = ""
        for item in data:
            all_words += item + " "

        tokens, vocab = BPE.bpe(all_words=all_words,
                                num_merges=num_merges,
                                vocab_size=vocab_size)
        vocab = BPE.update_vocab(vocab,
                                 tokens, [Tokens.PAD_TOKEN, Tokens.UNK_TOKEN])
        start = timeit.default_timer()
        debug("Init BPE tokenizing")
        # Sometimes we have puntuation chars in token
        # whose cannot be handled by regexp
        # I'm pretty confident this process can be achieved optimally
        tokens = {k.replace("[", "\["): v for k, v in tokens.items()}
        tokens = {k.replace("]", "\]"): v for k, v in tokens.items()}

        res = BPE.tokenize(data,
                           vocab,
                           tokens,
                           Tokens.UNK_TOKEN)

        dataset.representations["vocab"] = vocab

        debug(f"Finish BPE tokenizing after {timeit.default_timer() - start}")
        with Chained():
            data.loc[data.index] = res

        return dataset
