Three quick usage examples for these scripts:

### `run_pre_train.py`: Pre-train ZEN model from scratch or BERT model

```shell
python run_pre_train.py  \
    --pregenerated_data /path/to/pregenerated_data   \
    --bert_model /path/to/bert_model  \
    --do_lower_case  \
    --output_dir /path/to/output_dir   \
    --epochs 20  \
    --train_batch_size 128   \
    --fp16  \
    --scratch  \
    --save_name ZEN_pretrain_base_
```

### `run_sequence_level_ft_task.py`: Fine-tune on tasks for sequence classification

```shell
python run_sequence_level_ft_task.py \
    --task_name TASKNAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 512 \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 30.0
```
where TASKNAME can be one of DC, SA, SPM, NLI and QA

script of fine-tuning thucnews
```shell
python run_sequence_level_ft_task.py \
    --task_name thucnews \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset/thucnews \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 512 \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 30.0
```

script of fine-tuning chnsenticorp
```shell
python run_sequence_level_ft_task.py \
    --task_name ChnSentiCorp \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset/ChnSentiCorp \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 512 \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 30.0
```

script of fine-tuning LCQMC
```shell
python run_sequence_level_ft_task.py \
    --task_name lcqmc \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset/lcqmc \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 128 \
    --train_batch_size 128 \
    --learning_rate 5e-5 \
    --num_train_epochs 30.0
```

script of fine-tuning BQ-Corpus
```shell
python run_sequence_level_ft_task.py \
    --task_name bq \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset/lcqmc \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 128 \
    --train_batch_size 128 \
    --learning_rate 5e-5 \
    --num_train_epochs 30.0
```

script of fine-tuning XNLI
```shell
python run_sequence_level_ft_task.py \
    --task_name xnli \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset/xnli \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 128 \
    --train_batch_size 128 \
    --learning_rate 5e-5 \
    --num_train_epochs 30.0
```


### `run_token_level_ft_task.py`: Fine-tune on tasks for sequence classification

```shell
python run_token_level_ft_task.py \
    --task_name TASKNAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 128 \
    --do_train  \
    --do_eval \
    --train_batch_size 128 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1
```
where TASKNAME can be one of CWS, POS and NER

script of fine-tuning msra
```shell
python run_token_level_ft_task.py \
    --task_name cwsmsra \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 256 \
    --do_train  \
    --do_eval \
    --train_batch_size 96 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1
```

script of fine-tuning CTB5
```shell
python run_token_level_ft_task.py \
    --task_name pos \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 256 \
    --do_train  \
    --do_eval \
    --train_batch_size 96 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1
```

script of fine-tuning msra_ner
```shell
python run_token_level_ft_task.py \
    --task_name msra \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 128 \
    --do_train  \
    --do_eval \
    --train_batch_size 128 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1
```

### `run_mrc_task.py`: Fine-tune on tasks for MRC tasks

```shell
python run_mrc_task.py \
    --task_name TASKNAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file /path/to/train_file \
    --test_file /path/to/test_file \
    --predict_file /path/to/predict_file \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 512 \
    --doc_stride=128 \
    --do_train  \
    --do_eval \
    --train_batch_size 128 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1
```
where TASKNAME can be one of MRC

script of fine-tuning cmrc2018
```shell
python run_mrc_task.py \
    --task_name cmrc2018 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --init_checkpoint /path/to/zen_model \
    --max_seq_length 512 \
    --doc_stride=128 \
    --do_train  \
    --do_eval \
    --train_batch_size 96 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1
```