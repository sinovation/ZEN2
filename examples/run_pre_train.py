# coding: utf-8
# Copyright 2019 Sinovation Ventures AI Institute
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
"""PyTorch pretrain for ZEN model."""

from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
import time
import datetime
import h5py

from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from tensorboardX import SummaryWriter

from apex.optimizers import FusedLAMB
from apex import amp

from ZEN2 import WEIGHTS_NAME, CONFIG_NAME
from ZEN2 import ZenConfig, ZenForPreTraining
from ZEN2 import BertTokenizer, ZenNgramDict
from ZEN2 import PolyWarmUpScheduler

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length, max_seq_length=512, max_ngram_in_sequence=128):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.max_seq_length = max_seq_length
        self.max_ngram_in_sequence = max_ngram_in_sequence
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'ngram_ids', 'ngram_masks', 'ngram_starts', 'ngram_lengths', 'ngram_segment_ids',
                'ngram_freqs', 'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids,
         ngram_ids, ngram_masks, ngram_starts, ngram_lengths, ngram_segment_ids,
         ngram_freqs, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 11 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        #ngram
        max_seq_length = input_ids.size(0)
        max_ngram_in_sequence = ngram_ids.size(0)
        # ngram_ids_num = torch.nonzero(ngram_masks).size(0)
        ngram_ids_num = max_ngram_in_sequence
        ngram_padded_mask_indices = (ngram_masks == 0).nonzero()
        if len(ngram_padded_mask_indices) != 0:
            ngram_ids_num = ngram_padded_mask_indices[0].item()
        ngram_positions_matrix = np.zeros(shape=(max_seq_length, max_ngram_in_sequence), dtype=np.float)
        for i in range(ngram_ids_num):
            ngram_positions_matrix[ngram_starts[i]:ngram_starts[i] + ngram_lengths[i], i] = ngram_freqs[i]
        ngram_positions = torch.from_numpy(ngram_positions_matrix.astype(np.float))
        ngram_positions = torch.div(ngram_positions, torch.stack([torch.sum(ngram_positions, 1)] * ngram_positions.size(1)).t() + 1e-10)

        return [input_ids, segment_ids, input_mask, masked_lm_labels,
                ngram_ids, ngram_masks, ngram_positions, ngram_starts, ngram_lengths, ngram_segment_ids,
                next_sentence_labels]

def save_model(model, optimizer, tokenizer, ngram_dict, saving_path):
    if saving_path.is_dir() and list(saving_path.iterdir()):
        logging.warning(f"Output directory ({saving_path}) already exists and is not empty!")
    saving_path.mkdir(parents=True, exist_ok=True)

    logging.info("** ** * Saving pre-train model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    output_model_file = os.path.join(saving_path, WEIGHTS_NAME)
    output_config_file = os.path.join(saving_path, CONFIG_NAME)
    output_optimizer_file = os.path.join(saving_path, "optimizer.pt")

    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(optimizer.state_dict(), output_optimizer_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(saving_path)
    ngram_dict.save(saving_path)

def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument('--logfile', type=Path, default=None)
    parser.add_argument('--tmpdir', type=Path, default=None)
    parser.add_argument("--do_lower_case", action="store_true")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--scratch',
                        action='store_true',
                        help="Whether to train from scratch")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--save_name',
                        type=str,
                        default="zen",
                        help="The prefix used for saving the remote model")
    parser.add_argument("--save_step",
                        default=10000,
                        type=int)
    parser.add_argument("--max_step",
                        default=90000,
                        type=int)
    parser.add_argument("--already_trained_epoch",
                        default=0,
                        type=int)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--backend', type=str, default='nccl')

    args = parser.parse_args()


    if 'SLURM_NTASKS' in os.environ:
        args.world_size = int(os.environ['SLURM_NTASKS'])
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
    if 'SLURM_LOCALID' in os.environ:
        args.local_rank = int(os.environ['SLURM_LOCALID'])

    logfile_prefix = "ws_{ws}_btz_{btz}_lr_{lr}_warmup_{warmup}".format(
        ws=args.world_size,
        btz=args.train_batch_size,
        lr=args.learning_rate,
        warmup=args.warmup_proportion
    )
    logfile = None
    if args.local_rank == -1 or args.rank == 0:
        logfile = args.logfile / "log_pretrain_{}_{}".format(logfile_prefix, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    log_format = '%(asctime)-10s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=logfile, filemode='w')

    logging.info(args)
    logging.info("rank: {}, local_rank: {}, world_size: {}".format(args.rank, args.local_rank, args.world_size))

    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend=args.backend, init_method=args.init_method, rank=args.rank,
                                             world_size=args.world_size)
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
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

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    ngram_dict = ZenNgramDict(args.bert_model)

    if args.scratch:
        config = ZenConfig.from_json_file(os.path.join(args.bert_model, CONFIG_NAME))
        logging.info("Training from scratch, loadding config: {}".format(os.path.join(args.bert_model, CONFIG_NAME)))
        logging.info(config)
        model = ZenForPreTraining(config)
    else:
        model = ZenForPreTraining.from_pretrained(args.bert_model)

    if args.fp16:
        model.half()
    model.to(device)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    b1 = 0.9
    b2 = 0.999
    if args.train_batch_size * args.world_size * args.gradient_accumulation_steps > 256:
        b2 = 0.98
    logging.info("lr: {lr}, warmup: {warmup}, t_total: {t_total}, b1: {b1}, b2: {b2}".format(
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        t_total=args.max_step,
        b1=b1, b2=b2
    ))

    optimizer = FusedLAMB(optimizer_grouped_parameters,
                          lr=args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=args.warmup_proportion,
                                       total_steps=args.max_steps)
    if args.fp16:
        if args.loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=args.loss_scale)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    global_step = 0
    tr_loss = 0
    perplexity = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    logging.info("***** Running training *****")
    # logging.info("  Num examples = %d", total_train_examples)
    logging.info("  Batch size = %d", args.train_batch_size * args.world_size * args.gradient_accumulation_steps)
    logging.info("  Num steps = %d", args.max_step)
    model.train()

    using_SummaryWriter_for_debug = False
    if args.local_rank in [-1, 0] or args.world_size <= 1:
        using_SummaryWriter_for_debug = True
    if using_SummaryWriter_for_debug is True:
        tb_writer = SummaryWriter(logdir="runs_pretrain", filename_suffix=logfile_prefix)

    max_epoch_data = 0
    for epoch in range(args.epochs):
        epoch_pregenerated_data = args.pregenerated_data / f'epoch_{epoch}'
        if epoch_pregenerated_data.exists():
            max_epoch_data = epoch
            break
    if max_epoch_data <= 0:
        max_epoch_data = args.epochs

    for epoch in range(args.epochs):
        if epoch < args.already_trained_epoch:
            continue

        logging.info("epoch: {}".format(epoch))
        epoch_st = time.time()

        epoch_pregenerated_data = args.pregenerated_data / f'epoch_{epoch % max_epoch_data}'
        epoch_pregenerated_data_list = list(epoch_pregenerated_data.glob("*training*"))
        logging.info(epoch_pregenerated_data_list)
        if args.world_size > 1:
            while len(epoch_pregenerated_data_list) % args.world_size != 0:
                epoch_pregenerated_data_list += epoch_pregenerated_data_list[
                                                :args.world_size - len(epoch_pregenerated_data_list) % args.world_size]
            epoch_pregenerated_data_list = epoch_pregenerated_data_list[args.rank::args.world_size]
        logging.info(epoch_pregenerated_data_list)
        for idx, pregenerated_data_file in enumerate(epoch_pregenerated_data_list):
            epoch_dataset = pretraining_dataset(pregenerated_data_file, 80)

            train_sampler = RandomSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

            with tqdm(total=len(train_dataloader),
                      desc=f"Epoch {epoch} feature [{idx}] {pregenerated_data_file}",
                      disable=(args.local_rank not in [-1, 0] and args.world_size > 1)) as pbar:
                model.train()
                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    # input_ids, input_mask, segment_ids, lm_label_ids, is_next, ngram_ids, ngram_masks,  \
                    # ngram_positions, ngram_starts, ngram_lengths, ngram_segment_ids = batch
                    input_ids, segment_ids, input_mask, lm_label_ids, ngram_ids, ngram_masks, ngram_positions, \
                    ngram_starts, ngram_lengths, ngram_segment_ids, is_next = batch

                    loss = model(input_ids,
                                 ngram_ids,
                                 ngram_positions,
                                 segment_ids,
                                 ngram_segment_ids,
                                 input_mask,
                                 ngram_masks,
                                 lm_label_ids,
                                 is_next)

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    pbar.update(1)
                    mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                    perplexity = torch.exp(torch.tensor(mean_loss))
                    pbar.set_postfix_str(f"Loss: {mean_loss:.5f} ppl: {perplexity:.5f}")
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        lr_scheduler.step()  # learning rate warmup
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if using_SummaryWriter_for_debug is True:
                            tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
                            tb_writer.add_scalar('loss', loss.item(), global_step)
                            tb_writer.add_scalar('ppl', perplexity, global_step)
                        optimizer.step()
                        optimizer.zero_grad()
                        model.zero_grad()
                        global_step += 1

                        if (args.local_rank == -1 or torch.distributed.get_rank() == 0) and \
                                (global_step % args.save_step == 0 or global_step > args.max_step):
                            # Save a trained model
                            ts = time.time()
                            st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')

                            saving_path = "checkpoint_{}ws_{}_{}_step_{}".format(args.save_name, args.world_size, st, global_step)
                            saving_path = Path(os.path.join(args.output_dir, saving_path))
                            save_model(model, optimizer, tokenizer, ngram_dict, saving_path)

                        if global_step > args.max_step:
                            break

            logging.info("epoch: {}, feature index: {}, runtime: {}s, ppl: {}".format(epoch, idx, (time.time() - epoch_st), perplexity))

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Save a trained model
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')

            saving_path = "{}ws_{}_{}_epoch_{}".format(args.save_name, args.world_size, st, epoch)
            saving_path = Path(os.path.join(args.output_dir, saving_path))
            save_model(model, optimizer, tokenizer, ngram_dict, saving_path)



        logging.info("rank: {}, local_rank: {}, epoch: {}, runtime: {}s".format(
            args.rank, args.local_rank, epoch, (time.time() - epoch_st)))

        if global_step > args.max_step:
            break

    if using_SummaryWriter_for_debug is True:
        tb_writer.close()

if __name__ == '__main__':
    main()