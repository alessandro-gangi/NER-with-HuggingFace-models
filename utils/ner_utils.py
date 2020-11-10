import re
from pathlib import Path
#import pandas as pd
#from pandas._libs import json
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

"""
def read_dataset_as_dataframe(path, inline_sep, seq_sep):
    
    ASSUMPTION: dataset is in the form <word, tag> with a single word and tag/label per line.
    Read the dataset inside 'path' and extracts text and tags
    :param path: path to dataset
    :param inline_sep: separator between word and tag/label in a line
    :param seq_sep: regular expression to separate two sequences (sentences) in the dataset
    :return: a dataframe representing the dataset
    
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
"""
