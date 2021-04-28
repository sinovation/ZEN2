# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run MRC task on ZEN2 model."""

from __future__ import absolute_import, division, print_function

import re
import argparse
import collections
import json
import logging
import math
import os
import random
import time
import datetime
import sys
from io import open
import glob

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from apex import amp
from ZEN2 import LinearWarmUpScheduler
from ZEN2 import ZenForQuestionAnswering, ZenConfig, WEIGHTS_NAME, CONFIG_NAME, VOCAB_NAME
from ZEN2 import ZenNgramDict
from ZEN2 import BertAdam
from ZEN2 import (BasicTokenizer, BertTokenizer, whitespace_tokenize,
                  _is_whitespace, convert_to_unicode, _is_punctuation, _is_control)
import nltk


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
def is_main_process():
    return get_rank() == 0

# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = in_str.lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    #handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def remove_punctuation(in_str):
    in_str = in_str.lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p], mmax

def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision 	= 1.0*lcs_len/len(prediction_segs)
        recall 		= 1.0*lcs_len/len(ans_segs)
        f1 			= (2*precision*recall)/(precision+recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em

def mrc_evaluate(examples, predictions):
    f1 = exact_match = total = 0
    for example in examples:
        total += 1
        if example.qas_id not in predictions:
            message = 'Unanswered question ' + example.qas_id + \
                      ' will receive score 0.'
            print(message, file=sys.stderr)
            continue
        ground_truths = list(map(lambda x: x['text'], example.answers))
        prediction = predictions[example.qas_id]
        exact_match += calc_em_score(ground_truths, prediction)
        f1 += calc_f1_score(ground_truths, prediction)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

class MrcExample(object):
    """
    A single training/test example for the Mrc dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 answers=None,
                 paragraph_text=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.answers = answers
        self.paragraph_text=paragraph_text

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 input_span_mask,
                 ngram_ids,
                 ngram_positions,
                 ngram_lengths,
                 ngram_tuples,
                 ngram_seg_ids,
                 ngram_masks,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_span_mask = input_span_mask
        self.start_position = start_position
        self.end_position = end_position

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks

def customize_tokenizer(text, tokenizer):
  temp_x = ""
  text = convert_to_unicode(text)
  for c in text:
    if tokenizer._is_chinese_char(ord(c)) or _is_punctuation(c) or _is_whitespace(c) or _is_control(c):
      temp_x += " " + c + " "
    else:
      temp_x += c
  if tokenizer.do_lower_case:
    temp_x = temp_x.lower()
  return temp_x.split()

def read_mrc_examples(input_file, is_training, tokenizer):
    """Read a MRC json file into a list of MrcExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            # raw_doc_tokens = tokenizer.basic_tokenizer.tokenize(paragraph_text)
            raw_doc_tokens = customize_tokenizer(paragraph_text, tokenizer.basic_tokenizer)
            doc_tokens = []
            char_to_word_offset = []

            k = 0
            temp_word = ""
            for c in paragraph_text:
                if _is_whitespace(c):
                    char_to_word_offset.append(k-1)
                    continue
                else:
                    temp_word += c
                    char_to_word_offset.append(k)
                if tokenizer.basic_tokenizer.do_lower_case:
                    temp_word = temp_word.lower()
                if temp_word == raw_doc_tokens[k]:
                    doc_tokens.append(temp_word)
                    temp_word = ""
                    k += 1

            assert k == len(raw_doc_tokens)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None

                answers = qa["answers"]
                if is_training:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]

                    if orig_answer_text not in paragraph_text:
                        logger.warning("Could not find answer: '%s'", orig_answer_text)
                    else:
                        answer_offset = paragraph_text.index(orig_answer_text)
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = "".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = "".join(whitespace_tokenize(orig_answer_text))
                        if tokenizer.basic_tokenizer.do_lower_case:
                            cleaned_answer_text = cleaned_answer_text.lower()
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                            continue

                example = MrcExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    answers=answers,
                    paragraph_text=paragraph_text
                )
                examples.append(example)

    logging.info("**********read_mrc_examples complete!**********")

    return examples


def convert_examples_to_features(examples, tokenizer, ngram_dict, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            input_span_mask = []

            tokens.append("[CLS]")
            segment_ids.append(0)
            input_span_mask.append(1)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
                input_span_mask.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            input_span_mask.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
                input_span_mask.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            input_span_mask.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                input_span_mask.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(input_span_mask) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            # ----------- code for ngram BEGIN-----------
            ngram_matches = []
            #  Filter the word segment from 2 to max_ngram_len to check whether there is a word
            max_gram_n = ngram_dict.max_ngram_len
            for p in range(2, max_gram_n):
                for q in range(0, len(tokens) - p + 1):
                    character_segment = tokens[q:q + p]
                    # j is the starting position of the word
                    # i is the length of the current word
                    character_segment = tuple(character_segment)
                    if character_segment in ngram_dict.ngram_to_id_dict:
                        ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                        ngram_freq = ngram_dict.ngram_to_freq_dict[character_segment]
                        ngram_matches.append([ngram_index, q, p, character_segment, ngram_freq])

            # shuffle(ngram_matches)
            ngram_matches = sorted(ngram_matches, key=lambda s: s[0])
            # max_word_in_seq_proportion = max_word_in_seq
            max_word_in_seq_proportion = math.ceil((len(tokens) / max_seq_length) * ngram_dict.max_ngram_in_seq)
            if len(ngram_matches) > max_word_in_seq_proportion:
                ngram_matches = ngram_matches[:max_word_in_seq_proportion]
            ngram_ids = [ngram[0] for ngram in ngram_matches]
            ngram_positions = [ngram[1] for ngram in ngram_matches]
            ngram_lengths = [ngram[2] for ngram in ngram_matches]
            ngram_tuples = [ngram[3] for ngram in ngram_matches]
            ngram_freqs = [ngram[4] for ngram in ngram_matches]
            ngram_seg_ids = [0 if position < len([id for id in segment_ids if id==0]) else 1 for position in ngram_positions]

            ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
            ngram_mask_array[:len(ngram_ids)] = 1

            # record the masked positions
            ngram_positions_matrix = np.zeros(shape=(max_seq_length, ngram_dict.max_ngram_in_seq), dtype=np.int32)
            for i in range(len(ngram_ids)):
                ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = ngram_freqs[i]
            ngram_positions_matrix = torch.from_numpy(ngram_positions_matrix.astype(np.float))
            ngram_positions_matrix = torch.div(ngram_positions_matrix, torch.stack(
                [torch.sum(ngram_positions_matrix, 1)] * ngram_positions_matrix.size(1)).t() + 1e-10)
            ngram_positions_matrix = ngram_positions_matrix.numpy()

            # Zero-pad up to the max word in seq length.
            padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
            ngram_ids += padding
            ngram_lengths += padding
            ngram_seg_ids += padding

            # ----------- code for ngram END-----------

            if example_index < 2:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("input_span_mask: %s" % " ".join([str(x) for x in input_span_mask]))
                logger.info("ngram_ids: %s" % " ".join([str(x) for x in ngram_ids]))
                logger.info("ngram_positions: %s" % " ".join([str(x) for x in ngram_positions]))
                logger.info("ngram_lengths: %s" % " ".join([str(x) for x in ngram_lengths]))
                logger.info("ngram_tuples: %s" % " ".join([str(x) for x in ngram_tuples]))
                logger.info("ngram_seg_ids: %s" % " ".join([str(x) for x in ngram_seg_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info("answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    input_span_mask=input_span_mask,
                    start_position=start_position,
                    end_position=end_position,
                    ngram_ids=ngram_ids,
                    ngram_positions=ngram_positions_matrix,
                    ngram_lengths=ngram_lengths,
                    ngram_tuples=ngram_tuples,
                    ngram_seg_ids=ngram_seg_ids,
                    ngram_masks=ngram_mask_array
                ))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The MRC annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in MRC, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def search_answer_in_ori_text(text, paragraph_text, do_lower_case):
    if do_lower_case:
        paragraph_text = paragraph_text.lower()
    tokens = text.split(" ")
    text = text.replace(" ","")
    index = len(tokens)
    while True:
        tmp_text = ''.join(tokens[:index])
        if tmp_text not in paragraph_text:
            index -= 1
            if index < 1:
                return text
            continue
        if index == len(tokens):
            return "".join(tokens)
        if tmp_text+" " not in paragraph_text:
            return text
        tokens = tokens[:index] + [" "] + tokens[index:]
        index = len(tokens)
    return text

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, verbose_logging):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                # final_text = final_text.replace(' ', '')
                final_text = search_answer_in_ori_text(final_text, example.paragraph_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the MRC eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


from apex.multi_tensor_apply import multi_tensor_applier
class GradientClipper:
    """
    Clips gradient norm of an iterable of parameters.
    """
    def __init__(self, max_grad_norm):
        self.max_norm = max_grad_norm
        if multi_tensor_applier.available:
            import amp_C
            self._overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_scale = amp_C.multi_tensor_scale
        else:
            raise RuntimeError('Gradient clipping requires cuda extensions')

    def step(self, parameters):
        l = [p.grad for p in parameters if p.grad is not None]
        total_norm, _ = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [l], False)
        total_norm = total_norm.item()
        if (total_norm == float('inf')): return
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [l, l], clip_coef)

def save_zen_model(save_zen_model_path, model, args):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_zen_model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(save_zen_model_path, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    with open(output_config_file, "w", encoding='utf-8') as writer:
        writer.write(model_to_save.config.to_json_string())
    output_args_file = os.path.join(save_zen_model_path, 'training_args.bin')
    torch.save(args, output_args_file)

def train(args, model, tokenizer, ngram_dict, device, n_gpu):
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_mrc_examples(
            input_file=args.train_file, is_training=True, tokenizer=tokenizer)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.do_train:
        if args.fp16:
            try:
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            # import ipdb; ipdb.set_trace()
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False)

            if args.loss_scale == 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale="dynamic")
            else:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                  loss_scale=args.loss_scale)
            if args.do_train:
                scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                                  total_steps=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    # print(model)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            ngram_dict=ngram_dict
        )
        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

        all_ngram_ids = torch.tensor([f.ngram_ids for f in train_features], dtype=torch.long)
        all_ngram_positions = torch.tensor([f.ngram_positions for f in train_features], dtype=torch.long)
        all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
        all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
        all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions,
                                   all_ngram_ids, all_ngram_positions, all_ngram_lengths, all_ngram_seg_ids,
                                   all_ngram_masks)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        gradClipper = GradientClipper(max_grad_norm=1.0)
        for epoch_num in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                # Terminate early for benchmarking

                if args.max_steps > 0 and global_step > args.max_steps:
                    break

                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions, \
                ngram_ids, ngram_positions, ngram_lengths, ngram_seg_ids, ngram_masks = batch
                loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                             input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions,
                             start_positions=start_positions, end_positions=end_positions)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_steps += 1

                # gradient clipping
                gradClipper.step(amp.master_params(optimizer))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if step % args.log_freq == 0:
                    logger.info(
                        "Step {}: Loss {}, LR {} ".format(global_step, loss.item(), optimizer.param_groups[0]['lr']))

            if args.local_rank == -1 or torch.distributed.get_rank() == 0 or args.world_size <= 1:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "epoch-{}".format(epoch_num))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                save_zen_model(output_dir, model, args)

    loss = tr_loss / nb_tr_steps if args.do_train else None
    return loss, global_step


def evaluate(args, model, tokenizer, ngram_dict, device, mode='dev'):
    if args.fp16:
        model.half()

    if mode == "test":
        input_file=args.test_file
    else:
        input_file = args.predict_file
    eval_examples = read_mrc_examples(
        input_file=input_file, is_training=False, tokenizer=tokenizer)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False,
        ngram_dict=ngram_dict
    )

    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_ngram_ids = torch.tensor([f.ngram_ids for f in eval_features], dtype=torch.long)
    all_ngram_positions = torch.tensor([f.ngram_positions for f in eval_features], dtype=torch.long)
    all_ngram_lengths = torch.tensor([f.ngram_lengths for f in eval_features], dtype=torch.long)
    all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in eval_features], dtype=torch.long)
    all_ngram_masks = torch.tensor([f.ngram_masks for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index,
                    all_ngram_ids, all_ngram_positions, all_ngram_lengths, all_ngram_seg_ids, all_ngram_masks)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices, \
        ngram_ids, ngram_positions, ngram_lengths, ngram_seg_ids, ngram_masks \
            in tqdm(eval_dataloader, desc="Evaluating"):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        ngram_ids = ngram_ids.to(device)
        ngram_positions = ngram_positions.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids=input_ids,
                            token_type_ids=segment_ids, attention_mask=input_mask,
                             input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    predictions = write_predictions(eval_examples, eval_features, all_results,
                      args.n_best_size, args.max_answer_length,
                      args.do_lower_case, output_prediction_file,
                      output_nbest_file, args.verbose_logging)
    results = mrc_evaluate(eval_examples, predictions)
    print(results)
    return results

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="MRC json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="MRC json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--test_file", default=None, type=str,
                        help="MRC json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1.0, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal MRC evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--eval_all_cuda",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--vocab_file',
                        type=str, default=None,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument('--log_freq',
                        type=int, default=50,
                        help='frequency of logging loss.')
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="world size")
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--jobid', type=str, default='0')
    args = parser.parse_args()

    if 'SLURM_NTASKS' in os.environ:
        args.world_size = int(os.environ['SLURM_NTASKS'])
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
    if 'SLURM_LOCALID' in os.environ:
        args.local_rank = int(os.environ['SLURM_LOCALID'])
    if 'SLURM_JOBID' in os.environ:
        args.jobid = os.environ['SLURM_JOBID']

    if os.path.isfile(args.vocab_file) is False:
        args.vocab_file = os.path.join(args.init_checkpoint, VOCAB_NAME)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank,
                                             world_size=args.world_size)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    if args.do_train:
        str_args = "bts_{}_lr_{}_warmup_{}_seed_{}_jobid_{}".format(
            args.train_batch_size,
            args.learning_rate,
            args.warmup_proportion,
            args.seed,
            args.jobid
        )
        args.output_dir = os.path.join(args.output_dir, 'result-{}-{}-{}'.format(args.task_name, str_args, now_time))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        print("Output directory already exists and is not empty.")
    if not os.path.exists(args.output_dir) and (args.local_rank == -1 or args.world_size == 1 or torch.distributed.get_rank() == 0):
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case)
    ngram_dict = ZenNgramDict(args.init_checkpoint, tokenizer=tokenizer)
    model = ZenForQuestionAnswering.from_pretrained(args.init_checkpoint)

    if is_main_process():
        print("LOADED CHECKPOINT")
    model.to(device)

    results = {"init_checkpoint": args.init_checkpoint, "lr": args.learning_rate, "warmup": args.warmup_proportion,
               "train_batch_size": args.train_batch_size * args.gradient_accumulation_steps * args.world_size,
               "fp16": args.fp16}
    if args.do_train:
        results["train_start_runtime"] = time.time()
        loss, global_step = train(args, model, tokenizer, ngram_dict, device, n_gpu)
        results["train_runtime"] = time.time() - results["train_start_runtime"]
        results["global_step"] = global_step
        results["loss"] = loss
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        results["eval_start_runtime"] = time.time()
        if args.eval_all_cuda:
            args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        results["best_checkpoint"] = 0
        results["best_f1_score"] = 0
        results["best_em_score"] = 0
        results["best_checkpoint_path"] = ""
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = ZenForQuestionAnswering.from_pretrained(checkpoint)
            model.to(args.device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            result = evaluate(args, model, tokenizer, ngram_dict, device, mode='dev')
            if "exact_match" in result:
                if result["exact_match"] > results["best_em_score"]:
                    results["best_f1_score"] = result["f1"]
                    results["best_em_score"] = result["exact_match"]
                    results["best_checkpoint"] = global_step
                    results["best_checkpoint_path"] = checkpoint
            else:
                if result["f1"] > results["best_f1_score"]:
                    results["best_f1_score"] = result["f1"]
                    results["best_em_score"] = result["exact_match"]
                    results["best_checkpoint"] = global_step
                    results["best_checkpoint_path"] = checkpoint
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        if args.test_file is not None and os.path.exists(args.test_file):
            model = ZenForQuestionAnswering.from_pretrained(results["best_checkpoint_path"])
            model.to(args.device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            result = evaluate(args, model, tokenizer, ngram_dict, device, mode='test')
            result = {"test_{}".format(k): v for k, v in result.items()}
            results.update(result)
        results["eval_runtime"] = time.time() - results["eval_start_runtime"]
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write(json.dumps(results, ensure_ascii=False))
        for key in sorted(results.keys()):
            logger.info("{} = {}\n".format(key, str(results[key])))


if __name__ == "__main__":
    main()
