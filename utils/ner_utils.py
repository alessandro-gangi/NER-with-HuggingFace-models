import re
from pathlib import Path
import pandas as pd
from pandas._libs import json
from transformers import TrainingArguments


def read_dataset(path, inline_sep, seq_sep, read_labels=True):
    """
    ASSUMPTION: dataset is in the form <word, tag> with a single word and tag/label per line.
    Read the dataset inside 'path' and extracts text and tags
    :param path: path to dataset
    :param inline_sep: separator between word and tag/label in a line
    :param seq_sep: regular expression to separate two sequences (sentences) in the dataset
    :return: a list of sequences and a list of tags/labels
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding='utf-8').strip()
    raw_sequences = re.split(r'' + seq_sep, raw_text)

    text = []
    labels = []
    for raw_seq in raw_sequences:
        tmp_sent = []
        tmp_tags = []
        for raw_line in raw_seq.split('\n'):
            splits = raw_line.split(inline_sep)
            tmp_word = ''.join([i if ord(i) < 128 else '-' for i in splits[0]])  #replace non ASCII char
            tmp_sent.append(tmp_word)
            if read_labels:
                tmp_tag = splits[-1]
                tmp_tags.append(tmp_tag)

        text.append(tmp_sent)
        if read_labels:
            labels.append(tmp_tags)
            assert len(tmp_sent) == len(tmp_tags)

    return text, labels


def read_labels_list(path: str, sep='\n'):
    """
    ASSUMPTION: labels file contains one label per line
    :param path: str
        path to labels file
    :param sep: str
        regular expression tu split between lines
    :return labels: list
        a list of labels read from the file
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding='utf8').strip()
    labels = re.split(r''+sep, raw_text)

    return labels


def read_dataset_as_dataframe(path, inline_sep, seq_sep):
    """
    ASSUMPTION: dataset is in the form <word, tag> with a single word and tag/label per line.
    Read the dataset inside 'path' and extracts text and tags
    :param path: path to dataset
    :param inline_sep: separator between word and tag/label in a line
    :param seq_sep: regular expression to separate two sequences (sentences) in the dataset
    :return: a dataframe representing the dataset
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding='utf8').strip()
    raw_sequences = re.split(r'' + seq_sep, raw_text)

    data = []
    indexes = []
    seq_index = 0
    for raw_seq in raw_sequences:
        for raw_line in raw_seq.split('\n'):
            splits = raw_line.split(inline_sep)
            text = splits[0]
            label = splits[-1]
            data.append([text, label])
            indexes.append(seq_index)

        seq_index += 1

    return pd.DataFrame(data=data, index=indexes, columns=['text', 'label'])


def get_args_from_config(path: str = './config.json'):
    """
    Maps arguments inside the .json config file in arguments objects
    :param path: path to a .json config file
    :return: 3 arguments objects
    """
    with open(path) as handle:
        configdump = json.loads(handle.read())

    model_args = ModelArguments(
        model_name_or_path=configdump['model_args']['model_name_or_path'],  # model name ora path to saved model
        use_fast=configdump['model_args']['use_fast'],  # fast tokenizer with offset mapping (must be True)
        config_cache_dir=configdump['model_args']['config_cache_dir'],  # directory for caching models configurations
        model_cache_dir=configdump['model_args']['model_cache_dir'],  # directory for caching model configurations
        tokenizer_cache_dir=configdump['model_args']['tokenizer_cache_dir']  # directory for caching tokenizers
    )

    data_args = DataTrainingArguments(
        data_dir=configdump['data_args']['data_dir'],  # directory where datasets are located
        labels_dir=configdump['data_args']['labels_dir'],  # directory where labels.txt is located
        inline_sep=configdump['data_args']['inline_sep'],  # separator between word and tag inside a line
        seq_sep=configdump['data_args']['seq_sep'],  # separator between 2 sequences/sentences
        max_seq_length=configdump['data_args']['max_seq_length'],  # length to decide where to truncate and pad
        overwrite_cache=configdump['data_args']['overwrite_cache']  # flag to chose if overwrite the cached training
        # and evaluation sets
    )

    training_args = TrainingArguments(
        output_dir=configdump['training_args']['output_dir'],  # directory for saving model, tokenizer and results
        overwrite_output_dir=configdump['training_args']['overwrite_output_dir'],  # to overwrite mod, tok and results
        num_train_epochs=configdump['training_args']['num_train_epochs'],
        per_device_train_batch_size=configdump['training_args']['per_device_train_batch_size'],
        per_device_eval_batch_size=configdump['training_args']['per_device_eval_batch_size'],
        warmup_steps=configdump['training_args']['warmup_steps'],  # number of warmup steps for learning rate scheduler
        weight_decay=configdump['training_args']['weight_decay'],  # strength of weight decay
        logging_dir=configdump['training_args']['logging_dir'],  # directory for storing logs
        logging_steps=configdump['training_args']['logging_steps'],  # steps between one log and the next
        evaluate_during_training=configdump['training_args']['evaluate_during_training'],  # flag to evaluate at
        # every log
        do_train=configdump['training_args']['do_train'],  # enable training phase
        do_eval=configdump['training_args']['do_eval'],  # enable evaluation phase
        do_predict=configdump['training_args']['do_predict'],  # enable prediction phase
    )
    return model_args, data_args, training_args
