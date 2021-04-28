# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
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

"""Run sequence level classification task on ZEN model."""

from __future__ import absolute_import, division, print_function

import pickle
import argparse
import csv
import logging
import os
import random
import string
import sys
import json
import time
import glob
import json
import math
import datetime

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from apex import amp
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from ZEN2 import BertTokenizer, BertAdam, LinearWarmUpScheduler
from ZEN2 import ZenForSequenceClassification, ZenConfig
from ZEN2 import ZenNgramDict
from ZEN2 import WEIGHTS_NAME, CONFIG_NAME, NGRAM_DICT_NAME, VOCAB_NAME

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import torch.distributed as dist

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
def is_main_process():
    return get_rank() == 0

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def evaluate_mrr(preds):
    last_qid = None
    total_mrr = 0.0
    qnum = 0.0
    rank = 0.0
    correct = False
    for qid, score, label in preds:
        if qid != last_qid:
            rank = 0.0
            qnum += 1
            correct = False
            last_qid = qid

        rank += 1
        if not correct and label != 0:
            total_mrr += 1.0 / rank
            correct = True

    return total_mrr / qnum

def f1_mrr(preds, labels, pred_score=None, qids=None):
    f1 = f1_score(y_true=labels, y_pred=preds)
    mrr_preds = sorted(zip(qids, pred_score, labels), key=lambda elem: (elem[0], -elem[1]))
    mrr = evaluate_mrr(mrr_preds)
    return {
        "f1":f1,
        "mrr":mrr
    }

def compute_metrics(task_name, preds, labels, pred_score=None, qids=None):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "xnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "fudansmall":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "fudanlarge":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "conll":
        return acc_and_f1(preds, labels)
    elif task_name == "nerrenmin":
        return acc_and_f1(preds, labels)
    elif task_name == "nermsra":
        return acc_and_f1(preds, labels)
    elif task_name == "cwsmsra":
        return acc_and_f1(preds, labels)
    elif task_name == "cwspku":
        return acc_and_f1(preds, labels)
    elif task_name == "thucnews":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "chnsenticorp":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "lcqmc":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "bq":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "nlpccdbqa":
        return f1_mrr(preds, labels, pred_score, qids)
    else:
        raise KeyError(task_name)



def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, qid=0):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.qid = qid


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 ngram_ids, ngram_starts, ngram_lengths, ngram_tuples, ngram_seg_ids, ngram_masks, ngram_freqs,
                 qid=-1):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.qid = qid

        self.ngram_ids = ngram_ids
        self.ngram_starts = ngram_starts
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks
        self.ngram_freqs = ngram_freqs


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class XnliProcessor(DataProcessor):
    """Processor for the XNLI data set (HIT version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2].strip()
            if label == "contradictory":
                label = "contradiction"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class LcqmcProcessor(DataProcessor):
    """Processor for the XNLI data set (HIT version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2].strip()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class BqProcessor(DataProcessor):
    """Processor for the BQ Corpus data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "BQ_train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "BQ_dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "BQ_test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            label = line["gold_label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            lines = []
            for line in f:
                lines.append(json.loads(line.strip()))
            return lines

class ChnsenticorpProcessor(DataProcessor):
    """Processor for the XNLI data set (HIT version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            # text_a = line[0]
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

class ThucnewsProcessor(DataProcessor):
    """Processor for the XNLI data set (HIT version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['财经', '教育', '游戏', '科技', '时政', '时尚', '房产', '体育', '娱乐', '家居']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #    continue
            guid = "%s-%s" % (set_type, i)
            # text_a = line[0]
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

class NlpccDbqaProcessor(DataProcessor):
    """Processor for the NLPCC-DBQA data set (HIT version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
               continue
            guid = "%s-%s" % (set_type, i)
            qid = int(line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, qid=qid))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, ngram_dict):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

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
        ngram_seg_ids = [0 if position < len([id for id in segment_ids if id == 0]) else 1 for position in
                         ngram_positions]

        ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
        ngram_mask_array[:len(ngram_ids)] = 1

        # Zero-pad up to the max word in seq length.
        padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding
        ngram_positions += padding
        ngram_lengths += padding
        ngram_seg_ids += padding
        ngram_freqs += padding

        # ----------- code for ngram END-----------

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("ngram_ids: %s" % " ".join([str(x) for x in ngram_ids]))
            logger.info("ngram_positions: %s" % " ".join([str(x) for x in ngram_positions]))
            logger.info("ngram_lengths: %s" % " ".join([str(x) for x in ngram_lengths]))
            logger.info("ngram_tuples: %s" % " ".join([str(x) for x in ngram_tuples]))
            logger.info("ngram_seg_ids: %s" % " ".join([str(x) for x in ngram_seg_ids]))
            logger.info("ngram_freqs: %s" % " ".join([str(x) for x in ngram_freqs]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              ngram_ids=ngram_ids,
                              ngram_starts=ngram_positions,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array,
                              ngram_freqs=ngram_freqs,
                              qid=example.qid))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

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

def train(args, model, tokenizer, ngram_dict, processor, label_list, device, n_gpu):
    train_data = load_examples(args, tokenizer, ngram_dict, processor, label_list, "train")
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    num_train_optimization_steps = int(
        len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        print("using fp16")
        try:
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)

        if args.loss_scale == 0:

            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                              loss_scale=args.loss_scale)
        scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion,
                                          total_steps=num_train_optimization_steps)
    else:
        print("using fp32")
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

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

    model.train()
    for epoch_num in trange(int(args.num_train_epochs), desc="Epoch"):
        if args.max_steps > 0 and global_step > args.max_steps:
            break
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if args.max_steps > 0 and global_step > args.max_steps:
                break
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, qids, ngram_ids, ngram_starts, \
            ngram_lengths, ngram_seg_ids, ngram_masks, ngram_freqs = batch

            batch_size = ngram_ids.size(0)
            ngram_positions_matrix = torch.zeros(size=(batch_size, args.max_seq_length, ngram_dict.max_ngram_in_seq),
                                                 dtype=torch.int, device=device)
            for batch_id in range(batch_size):
                ngram_id = ngram_ids[batch_id]
                ngram_start = ngram_starts[batch_id]
                ngram_length = ngram_lengths[batch_id]
                for i in range(len(ngram_id)):
                    ngram_positions_matrix[batch_id][ngram_start[i]:ngram_start[i] + ngram_length[i], i] = ngram_freqs[batch_id][i]
                ngram_positions_matrix[batch_id] \
                    = torch.div(ngram_positions_matrix[batch_id],
                                torch.stack([torch.sum(ngram_positions_matrix[batch_id], 1)] *
                                            ngram_positions_matrix[batch_id].size(1)).t() + 1e-10)


            loss = model(input_ids=input_ids,
                         input_ngram_ids=ngram_ids,
                         ngram_position_matrix=ngram_positions_matrix,
                         attention_mask=input_mask,
                         token_type_ids=segment_ids,
                         labels=label_ids)
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
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up for BERT which FusedAdam doesn't do
                    scheduler.step()

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        if args.local_rank == -1 or torch.distributed.get_rank() == 0 or args.world_size <= 1:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "epoch-{}".format(epoch_num))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_zen_model(output_dir, model, args)

    loss = tr_loss / nb_tr_steps if args.do_train else None
    return loss, global_step

def load_examples(args, tokenizer, ngram_dict, processor, label_list, mode):
    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)

    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, ngram_dict)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_qids = torch.tensor([f.qid for f in features], dtype=torch.long)

    all_ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long)
    all_ngram_starts = torch.tensor([f.ngram_starts for f in features], dtype=torch.long)
    all_ngram_lengths = torch.tensor([f.ngram_lengths for f in features], dtype=torch.long)
    all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in features], dtype=torch.long)
    all_ngram_masks = torch.tensor([f.ngram_masks for f in features], dtype=torch.long)
    all_ngram_freqs = torch.tensor([f.ngram_freqs for f in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_qids,
                         all_ngram_ids, all_ngram_starts, all_ngram_lengths, all_ngram_seg_ids, all_ngram_masks, all_ngram_freqs)

def evaluate(args, model, tokenizer, ngram_dict, processor, label_list, device, mode="test"):
    eval_data = load_examples(args, tokenizer, ngram_dict, processor, label_list, mode)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    pred_scores = None
    out_label_ids = None
    q_ids = None
    for input_ids, input_mask, segment_ids, label_ids, qids, input_ngram_ids, ngram_starts, \
        ngram_lengths, ngram_seg_ids, ngram_masks, ngram_freqs in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        input_ngram_ids = input_ngram_ids.to(device)
        ngram_starts = ngram_starts.to(device)

        batch_size = input_ngram_ids.size(0)
        ngram_positions_matrix = torch.zeros(size=(batch_size, args.max_seq_length, ngram_dict.max_ngram_in_seq),
                                             dtype=torch.int, device=device)
        for batch_id in range(batch_size):
            ngram_id = input_ngram_ids[batch_id]
            ngram_start = ngram_starts[batch_id]
            ngram_length = ngram_lengths[batch_id]
            for i in range(len(ngram_id)):
                ngram_positions_matrix[batch_id][ngram_start[i]:ngram_start[i] + ngram_length[i], i] = ngram_freqs[batch_id][i]
            ngram_positions_matrix[batch_id] \
                = torch.div(ngram_positions_matrix[batch_id],
                            torch.stack([torch.sum(ngram_positions_matrix[batch_id], 1)] *
                                        ngram_positions_matrix[batch_id].size(1)).t() + 1e-10)

        with torch.no_grad():
            logits = model(input_ids=input_ids,
                           input_ngram_ids=input_ngram_ids,
                           ngram_position_matrix=ngram_positions_matrix,
                           attention_mask=input_mask,
                           token_type_ids=segment_ids)
        nb_eval_steps += 1
        if pred_scores is None:
            pred_scores = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
            q_ids = qids.numpy().reshape(-1).tolist()
        else:
            pred_scores = np.append(pred_scores, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)
            q_ids = np.append(q_ids, qids.numpy().reshape(-1).tolist(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(pred_scores, axis=1)
    pred_scores = pred_scores[:, 1].reshape(-1).tolist()

    eval_loss = eval_loss / nb_eval_steps
    result = compute_metrics(args.task_name, preds, out_label_ids, pred_scores, q_ids)
    result["eval_loss"] = eval_loss

    return result

def predict(args, model, tokenizer, ngram_dict, processor, label_list, device, mode="test"):
    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)

    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, ngram_dict)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_qids = torch.tensor([f.qid for f in features], dtype=torch.long)

    all_ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long)
    all_ngram_starts = torch.tensor([f.ngram_starts for f in features], dtype=torch.long)
    all_ngram_lengths = torch.tensor([f.ngram_lengths for f in features], dtype=torch.long)
    all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in features], dtype=torch.long)
    all_ngram_masks = torch.tensor([f.ngram_masks for f in features], dtype=torch.long)
    all_ngram_freqs = torch.tensor([f.ngram_freqs for f in features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_qids,
                              all_ngram_ids, all_ngram_starts, all_ngram_lengths, all_ngram_seg_ids, all_ngram_masks, all_ngram_freqs)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    id2label_map = {i: label  for i, label in enumerate(label_list)}

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    pred_scores = None
    out_label_ids = None
    q_ids = None
    ngram_attentions = None
    for input_ids, input_mask, segment_ids, label_ids, qids, input_ngram_ids, ngram_starts, \
        ngram_lengths, ngram_seg_ids, ngram_masks, ngram_freqs in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        input_ngram_ids = input_ngram_ids.to(device)
        ngram_starts = ngram_starts.to(device)

        batch_size = input_ngram_ids.size(0)
        ngram_positions_matrix = torch.zeros(size=(batch_size, args.max_seq_length, ngram_dict.max_ngram_in_seq),
                                             dtype=torch.int, device=device)
        for batch_id in range(batch_size):
            ngram_id = input_ngram_ids[batch_id]
            ngram_start = ngram_starts[batch_id]
            ngram_length = ngram_lengths[batch_id]
            for i in range(len(ngram_id)):
                ngram_positions_matrix[batch_id][ngram_start[i]:ngram_start[i] + ngram_length[i], i] = \
                    ngram_freqs[batch_id][i]
            ngram_positions_matrix[batch_id] \
                = torch.div(ngram_positions_matrix[batch_id],
                            torch.stack([torch.sum(ngram_positions_matrix[batch_id], 1)] *
                                        ngram_positions_matrix[batch_id].size(1)).t() + 1e-10)

        with torch.no_grad():
            logits = model(input_ids=input_ids,
                           input_ngram_ids=input_ngram_ids,
                           ngram_position_matrix=ngram_positions_matrix,
                           attention_mask=input_mask,
                           token_type_ids=segment_ids)
            if args.output_attentions:
                attentions, logits = logits
        nb_eval_steps += 1
        if pred_scores is None:
            pred_scores = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
            q_ids = qids.numpy().reshape(-1).tolist()
            if args.output_attentions:
                ngram_attentions = attentions[0].detach().cpu().numpy()
            else:
                ngram_attentions = [None]*batch_size
        else:
            pred_scores = np.append(pred_scores, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)
            q_ids = np.append(q_ids, qids.numpy().reshape(-1).tolist(), axis=0)
            if args.output_attentions:
                ngram_attentions = np.append(ngram_attentions, attentions[0].detach().cpu().numpy(), axis=0)
            else:
                ngram_attentions.extend([None]*batch_size)

    preds = np.argmax(pred_scores, axis=1)

    with open(os.path.join(args.output_dir,"pred.txt"), 'w') as f:
        for example, feature, pred, ngram_attention in zip(examples, features, preds, ngram_attentions):
            ngram_num = len(feature.ngram_tuples)
            header_size = len(ngram_attention)
            atts = [0]*ngram_num
            for h in range(header_size):
                for i in range(ngram_num):
                    for j in range(ngram_num):
                        atts[j] += ngram_attention[h][i][j]
            sum_att = sum(atts)
            for i in range(ngram_num):
                atts[i] = atts[i] / sum_att
            print("{}\t{}\t{}\t{}\t{}\t{}".format(example.text_a,example.text_b, ",".join(["".join(t) for t in feature.ngram_tuples]),
                                                    example.label, id2label_map[pred], ",".join([str(x) for x in atts])))
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(example.text_a,example.text_b, ",".join(["".join(t) for t in feature.ngram_tuples]),
                                                    example.label, id2label_map[pred], ",".join([str(x) for x in atts])))

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_dev",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_all_cuda",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--google_pretrained",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--max_steps", default=-1.0, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--old", action='store_true', help="use old fp16 optimizer")
    parser.add_argument('--vocab_file',
                        type=str, default=None,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--jobid', type=str, default='0')
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--output_attentions",
                        action='store_true',
                        help="Whether to run training.")

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

    task_name = args.task_name.lower()
    args.task_name = args.task_name.lower()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
        "thucnews": ThucnewsProcessor,
        "chnsenticorp": ChnsenticorpProcessor,
        "lcqmc": LcqmcProcessor,
        "bq": BqProcessor,
        "nlpccdbqa": NlpccDbqaProcessor,
    }

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
    args.device = device
    args.n_gpu = n_gpu
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

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

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
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case)
    ngram_dict = ZenNgramDict(args.init_checkpoint, tokenizer=tokenizer)
    model = ZenForSequenceClassification.from_pretrained(args.init_checkpoint,
                                                         num_labels=num_labels,
                                                         output_attentions=args.output_attentions)

    model.to(device)

    results = {"init_checkpoint": args.init_checkpoint, "lr": args.learning_rate, "warmup": args.warmup_proportion,
               "train_batch_size": args.train_batch_size * args.gradient_accumulation_steps * args.world_size,
               "fp16": args.fp16}
    if args.do_train:
        results["train_start_runtime"] = time.time()
        loss, global_step = train(args, model, tokenizer, ngram_dict, processor, label_list, device, n_gpu)
        results["train_runtime"] = time.time() - results["train_start_runtime"]
        results["global_step"] = global_step
        results["loss"] = loss
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        results["eval_start_runtime"] = time.time()
        if args.eval_all_cuda:
            args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        results["best_checkpoint"] = 0
        results["best_acc_score"] = 0
        results["best_f1_score"] = 0
        results["best_mrr_score"] = 0
        results["best_checkpoint_path"] = ""
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = ZenForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
            model.to(args.device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            result = evaluate(args, model, tokenizer, ngram_dict, processor, label_list, device, mode="dev")
            if "acc" in result:
                if result["acc"] > results["best_acc_score"]:
                    results["best_acc_score"] = result["acc"]
                    results["best_checkpoint"] = global_step
                    results["best_checkpoint_path"] = checkpoint
                    results["dev_acc"] = result["acc"]
            else:
                if result["f1"] > results["best_f1_score"]:
                    results["best_f1_score"] = result["f1"]
                    results["best_checkpoint"] = global_step
                    results["best_checkpoint_path"] = checkpoint
                    results["best_mrr_score"] = result.get("mrr", 0)
                    results["dev_f1"] = result["f1"]
                    results["dev_mrr"] = result.get("mrr", 0)
            if global_step:
                result = {"{}_dev_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
            if args.eval_dev:
                result = evaluate(args, model, tokenizer, ngram_dict, processor, label_list, device, mode="test")
                if global_step:
                    results.update({"{}_test_{}".format(global_step, k): v for k, v in result.items()})
                if results["best_checkpoint_path"] == checkpoint:
                    results.update({"test_{}".format(k): v for k, v in result.items()})
        results["eval_runtime"] = time.time() - results["eval_start_runtime"]
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write(json.dumps(results, ensure_ascii=False))
        for key in sorted(results.keys()):
            logger.info("{} = {}\n".format(key, str(results[key])))

    if args.do_predict:
        predict(args, model, tokenizer, ngram_dict, processor, label_list, device, mode="test")

if __name__ == "__main__":
    main()
