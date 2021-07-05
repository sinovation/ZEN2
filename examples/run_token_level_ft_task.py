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

"""Run token level classification task on ZEN model."""

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
import re

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.distributed as dist

from apex import amp
from seqeval.metrics import classification_report, f1_score

from ZEN2 import BertTokenizer, BertAdam, LinearWarmUpScheduler
from ZEN2 import ZenForTokenClassification, ZenConfig
from ZEN2 import WEIGHTS_NAME, CONFIG_NAME, NGRAM_DICT_NAME, ZenNgramDict,VOCAB_NAME

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
def is_main_process():
    return get_rank() == 0

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ngram_ids, ngram_positions, ngram_lengths,
                 ngram_tuples, ngram_seg_ids, ngram_masks, valid_ids=None, label_mask=None, b_use_valid_filter=False):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks

        self.b_use_valid_filter = b_use_valid_filter

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
        '''
        read file
        return format :
        [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
        '''
        f = open(input_file)
        data = []
        sentence = []
        label = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.split('\t')
            sentence.append(splits[0])
            label.append(splits[-1][:-1])

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data



class PosProcessor(DataProcessor):
    """Processor for the cws POS CTB5 data set."""

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
        return ['NR', 'NN', 'CC', 'VV', 'NT', 'PU', 'LC', 'AS', 'ETC', 'DEC', 'CD', 'M', 'DEG', 'JJ', 'VC', 'AD', 'P',
                'PN', 'VA', 'DEV', 'DT', 'SB', 'OD', 'VE', 'CS', 'MSP', 'BA', 'FW', 'LB', 'NP', 'DER', 'SP', 'IJ', 'X',
                'VP', "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class GeniaProcessor(DataProcessor):
    """Processor for the Genia data set."""

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
        return ['B-G#protein_domain_or_region', 'I-G#cell_component', 'I-G#atom', 'B-G#other_artificial_source',
                'I-G#DNA_family_or_group', 'I-G#protein_molecule', 'promoter', 'I-G#nucleotide',
                'B-G#DNA_domain_or_region',
                'B-G#amino_acid_monomer', 'B-G#DNA_substructure', 'I-G#polynucleotide', 'B-G#protein_molecule',
                'B-G#other_organic_compound', 'I-G#tissue', 'B-G#mono_cell', 'I-G#RNA_N/A', 'B-G#inorganic',
                'I-G#protein_domain_or_region', 'B-G#nucleotide', 'I-G#inorganic', 'I-G#DNA_substructure',
                'B-G#DNA_molecule',
                'I-G#DNA_molecule', 'I-G#protein_substructure', 'B-G#other_name', 'I-G#other_organic_compound',
                'I-G#RNA_domain_or_region', 'I-G#RNA_molecule', 'B-G#RNA_family_or_group', 'I-G#cell_line',
                'B-G#polynucleotide',
                'I-G#peptide', 'B-G#virus', 'I-G#cell_type', 'B-G#atom', 'B-G#DNA_N/A', 'I-G#carbohydrate',
                'I-G#protein_complex',
                'B-G#cell_type', 'I-G#DNA_domain_or_region', 'B-G#cell_component', 'B-G#protein_family_or_group',
                'I-G#multi_cell',
                'I-G#body_part', 'B-G#cell_line', 'I-G#lipid', 'I-G#other_artificial_source',
                'B-G#RNA_domain_or_region',
                'B-G#protein_N/A', 'B-G#tissue', 'B-G#RNA_molecule', 'B-G#multi_cell', 'B-G#DNA_family_or_group',
                'B-G#protein_subunit', 'I-G#protein_N/A', 'I-G#RNA_family_or_group', 'B-G#body_part', 'B-G#peptide',
                'I-G#other_name', 'I-G#virus', 'I-G#protein_subunit', 'B-G#lipid', 'B-G#protein_substructure',
                'I-G#DNA_N/A',
                'B-G#protein_complex', 'I-G#protein_family_or_group', 'B-G#RNA_N/A', 'O', 'B-G#carbohydrate',
                'I-G#amino_acid_monomer', 'I-G#mono_cell', "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class CwsmsraProcessor(DataProcessor):
    """Processor for the cws msra data set."""

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
        return ["B", "I", "E", "S", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class PeopledailyProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def read_tsv(cls, input_file):
        '''
        read file
        return format :
        [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
        '''
        f = open(input_file)
        data = []
        sentence = []
        label = []
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[1])

        if len(sentence) > 0:
            data.append((sentence, label))
        return data

class NermsraProcessor(DataProcessor):
    """Processor for the msra-ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def read_tsv(self, input_file):
        '''
        read file
        return format :
        '''
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                splits = line.split('\t')
                if len(splits) != 2:
                    continue
                data.append([re.split(' |\x02', splits[0]), re.split(' |\x02', splits[1])])
        return data[1:]

class ConllProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

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
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, ngram_dict):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    label_map["[PAD]"] = 0

    features = []
    b_use_valid_filter = False
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            if len(tokens) + len(token) > max_seq_length - 2:
                break
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
                    b_use_valid_filter = True
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        # ----------- code for ngram BEGIN-----------
        ngram_matches = []
        #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
        max_gram_n = ngram_dict.max_ngram_len
        for p in range(2, max_gram_n):
            for q in range(0, len(tokens) - p + 1):
                character_segment = tokens[q:q + p]
                # j is the starting position of the ngram
                # i is the length of the current ngram
                character_segment = tuple(character_segment)
                if character_segment in ngram_dict.ngram_to_id_dict:
                    ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                    ngram_freq = ngram_dict.ngram_to_freq_dict[character_segment]
                    ngram_matches.append([ngram_index, q, p, character_segment, ngram_freq])

        ngram_matches = sorted(ngram_matches, key=lambda s: s[0])

        max_ngram_in_seq_proportion = math.ceil((len(tokens) / max_seq_length) * ngram_dict.max_ngram_in_seq)
        if len(ngram_matches) > max_ngram_in_seq_proportion:
            ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

        ngram_ids = [ngram[0] for ngram in ngram_matches]
        ngram_positions = [ngram[1] for ngram in ngram_matches]
        ngram_lengths = [ngram[2] for ngram in ngram_matches]
        ngram_tuples = [ngram[3] for ngram in ngram_matches]
        ngram_freqs = [ngram[4] for ngram in ngram_matches]
        ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

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

        # Zero-pad up to the max ngram in seq length.
        padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding
        ngram_lengths += padding
        ngram_seg_ids += padding

        # ----------- code for ngram END-----------

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (",".join([str(x) for x in example.label]), ",".join([str(x) for x in label_ids])))
            logger.info("valid: %s" % " ".join([str(x) for x in valid]))
            logger.info("b_use_valid_filter: %s" % str(b_use_valid_filter))
            logger.info("ngram_ids: %s" % " ".join([str(x) for x in ngram_ids]))
            logger.info("ngram_positions: %s" % " ".join([str(x) for x in ngram_positions]))
            logger.info("ngram_lengths: %s" % " ".join([str(x) for x in ngram_lengths]))
            logger.info("ngram_tuples: %s" % " ".join([str(x) for x in ngram_tuples]))
            logger.info("ngram_seg_ids: %s" % " ".join([str(x) for x in ngram_seg_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          ngram_ids=ngram_ids,
                          ngram_positions=ngram_positions_matrix,
                          ngram_lengths=ngram_lengths,
                          ngram_tuples=ngram_tuples,
                          ngram_seg_ids=ngram_seg_ids,
                          ngram_masks=ngram_mask_array,
                          valid_ids=valid,
                          label_mask=label_mask,
                          b_use_valid_filter=b_use_valid_filter))
    return features

def cws_evaluate_word_PRF(y_pred, y):
    #dict = {'E': 2, 'S': 3, 'B':0, 'I':1}
    cor_num = 0
    yp_wordnum = y_pred.count('E')+y_pred.count('S')
    yt_wordnum = y.count('E')+y.count('S')
    start = 0
    for i in range(len(y)):
        if y[i] == 'E' or y[i] == 'S':
            flag = True
            for j in range(start, i+1):
                if y[j] != y_pred[j]:
                    flag = False
            if flag:
                cor_num += 1
            start = i+1

    P = cor_num / float(yp_wordnum)
    R = cor_num / float(yt_wordnum)
    F = 2 * P * R / (P + R)
    print('Precision: ', P)
    print('Recall: ', R)
    print('F1-score: ', F)
    return {
        "precision":P,
        "recall":R,
        "f1":F
    }

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
            input_ids, input_mask, segment_ids, label_ids, ngram_ids, ngram_positions, ngram_lengths, \
            ngram_seg_ids, ngram_masks, valid_ids, l_mask, b_use_valid_filter = batch
            b_use_valid_filter = b_use_valid_filter.detach().cpu().numpy()[0]
            loss = model(input_ids,
                         attention_mask=input_mask,
                         token_type_ids=segment_ids,
                         labels=label_ids, valid_ids=valid_ids,
                         input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions,
                         b_use_valid_filter=b_use_valid_filter)
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
    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)

    all_ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long)
    all_ngram_positions = torch.tensor([f.ngram_positions for f in features], dtype=torch.long)
    all_ngram_lengths = torch.tensor([f.ngram_lengths for f in features], dtype=torch.long)
    all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in features], dtype=torch.long)
    all_ngram_masks = torch.tensor([f.ngram_masks for f in features], dtype=torch.long)

    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    all_b_use_valid_filter = torch.tensor([f.b_use_valid_filter for f in features], dtype=torch.bool)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ngram_ids,
                         all_ngram_positions, all_ngram_lengths, all_ngram_seg_ids, all_ngram_masks,
                         all_valid_ids, all_label_mask, all_b_use_valid_filter)

def evaluate(args, model, tokenizer, ngram_dict, processor, label_list, device, mode="test"):
    num_labels = len(label_list) + 1
    eval_data = load_examples(args, tokenizer, ngram_dict, processor, label_list, mode)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    label_map[0] = "[PAD]"
    nb_tr_examples, nb_tr_steps = 0, 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, ngram_ids, ngram_positions, \
        ngram_lengths, ngram_seg_ids, ngram_masks, valid_ids, l_mask, b_use_valid_filter = batch
        b_use_valid_filter = b_use_valid_filter.detach().cpu().numpy()[0]

        with torch.no_grad():
            logits = model(input_ids,
                           attention_mask=input_mask,
                           token_type_ids=segment_ids,
                           valid_ids=valid_ids,
                           input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions,
                           b_use_valid_filter=b_use_valid_filter)

        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1:
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])

    logger.info("nb_tr_examples: {}, nb_tr_steps: {}".format(nb_tr_examples, nb_tr_steps))

    if args.task_name == 'cwsmsra' or args.task_name == 'cwspku':
        # evaluating CWS
        y_true_all = []
        y_pred_all = []
        for y_true_item in y_true:
            y_true_all += y_true_item
        for y_pred_item in y_pred:
            y_pred_all += y_pred_item
        result = cws_evaluate_word_PRF(y_pred_all, y_true_all)
        logger.info("=======entity level========")
        logger.info("\n%s", ', '.join("%s: %s" % (key, val) for key, val in result.items()))
        logger.info("=======entity level========")
    else:
        # evaluating NER, POS
        report = classification_report(y_true, y_pred, digits=4)
        f = f1_score(y_true, y_pred)
        result = {"report": report, "f1": f}
        logger.info("=======entity level========")
        logger.info(report)
        logger.info("=======entity level========")

    return result

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
        "conll": ConllProcessor,
        "peopledaily": PeopledailyProcessor,
        "msra": PeopledailyProcessor,
        "nermsra": NermsraProcessor,
        "nerpd": PeopledailyProcessor,
        "cwsmsra": CwsmsraProcessor,
        "cwspku": CwsmsraProcessor,
        "genia": GeniaProcessor,
        "pos": PosProcessor
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

    if not args.do_train and not args.do_eval:
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
    num_labels = len(label_list)+1

    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case)
    ngram_dict = ZenNgramDict(args.init_checkpoint, tokenizer=tokenizer)
    model = ZenForTokenClassification.from_pretrained(args.init_checkpoint, num_labels=num_labels)

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
        results["best_f1_score"] = 0
        results["best_checkpoint_path"] = ""
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = ZenForTokenClassification.from_pretrained(checkpoint, num_labels=num_labels)
            model.to(args.device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            result = evaluate(args, model, tokenizer, ngram_dict, processor, label_list, device, mode="dev")
            if result["f1"] > results["best_f1_score"]:
                results["best_f1_score"] = result["f1"]
                results["best_checkpoint"] = global_step
                results["best_checkpoint_path"] = checkpoint
                results["dev_f1"] = result["f1"]
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

if __name__ == "__main__":
    main()
