import json
import random
import warnings

from spacy.lang.it import Italian
from spacy.gold import biluo_tags_from_offsets
import re
from pathlib import Path
from config import ENTITIES_AGGREGATION, ENTITIES_TO_DELETE
#import pandas as pd
#from pandas._libs import json


def read_dataset(path, data_format='doccano', split=(0.8, 0.2), seed=42, prep_entities=False):
    """
    Read a dataset as list of sequences/sentences and labels
    :param path: str
        Path of dataset
    :param data_format: str
        Format of the dataset: conll and doccano are currently supported
    :param split: tuple of length 2
        Representing the train-eval split percentages. Default (0.8, 0.2)
    :param seed: int used for random shuffling
        Set to None for random seed. Default 42 (for reproducibility)
    :param prep_entities: boolean
        If set, entities will be preprocessed (merged, removed) according to the config file
    :return:
    """
    text, labels = None, None
    if data_format == 'doccano':
        text, labels = read_doccano_dataset(path)
    elif data_format == 'conll':
        text, labels = read_conll_dataset(path)

    # shuffle text and labels
    seed = seed if seed else random.random()
    random.seed(seed)
    random.shuffle(text)
    random.seed(seed)
    random.shuffle(labels)

    # split data in training set and evaluation set
    train_perc, eval_perc = split
    train_size = int(round(train_perc*len(text)))
    train_text, train_labels = text[:train_size], labels[:train_size]
    eval_text, eval_labels = text[train_size:], labels[train_size:]

    if prep_entities:
        train_labels = preprocess_entities(train_labels, ENTITIES_AGGREGATION, ENTITIES_TO_DELETE)
        eval_labels = preprocess_entities(eval_labels, ENTITIES_AGGREGATION, ENTITIES_TO_DELETE)

    return train_text, train_labels, eval_text, eval_labels

def read_doccano_dataset(path):
    """
    Read the doccano dataset as list of sequences/sentences and labels
    :param path: str
        Path of dataset
    :return: (list, list)
        List of sequences and list of corresponding labels
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding='utf8').strip()

    json_strings = raw_text.split('\n')

    texts_ids, texts, labels = [], [], []
    for json_str in json_strings:
        json_obj = json.loads(json_str)

        texts_ids.append(json_obj['id'])
        texts.append(json_obj['text'])
        #labels.append([(lab[0], lab[1], lab[2]) for lab in json_obj['labels']])

        # Fix annotation ending span problems
        labels.append([(lab[0], lab[1], lab[2]) if ' ' != json_obj['text'][lab[1]-1] else (lab[0], lab[1]-1, lab[2])
                      for lab in json_obj['labels']])

        # Fix annotation starting span problems
        # labels.append([(lab[0], lab[1], lab[2]) if ' ' != json_obj['text'][lab[0]] else (lab[0]+1, lab[1], lab[2])
        #               for lab in json_obj['labels']])

    # Spacy gold tokenizer to tokenize text and tags tokens with provided tags
    nlp = Italian()
    output_texts = []
    output_tags = []

    warnings.filterwarnings("ignore", message=r"\[W030\]", category=UserWarning)
    cnt = 0  # test misalignment problems
    for i, text in enumerate(texts):
        offsets = labels[i]
        doc = nlp(text)
        tokenized_text = [token.text for token in doc]

        tags = biluo_tags_from_offsets(doc, offsets)

        # Delete ' ' tokens (blank space tokens) and relative tags
        for j, token in enumerate(tokenized_text):
            if token.isspace():
                tokenized_text.pop(j)
                tags.pop(j)

        # Replace U-TAGS AND L-TAGS (tu use IOB tagging format)
        # and check for corrupted alignment
        iob_tags = [t.replace('U-', 'B-') for t in tags]
        iob_tags = [t.replace('L-', 'I-') for t in iob_tags]

        # Replace misaligned tag '-' with 'O' tag
        iob_tags = [tag if tag != '-' else 'O' for tag in iob_tags]

        # test misalignment problems
        # for t in iob_tags:
        #     if t == '-':
        #        cnt+=1

        output_texts.append(tokenized_text)
        output_tags.append(iob_tags)
    # print(f'COUNT: {cnt}')

    return output_texts, output_tags


def read_conll_dataset(path):
    """
    Read the conll dataset as list of sequences/sentences and labels
    :param path: str
        Path of dataset
    :return: (list, list)
        List of sequences and list of corresponding labels
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding='utf-8').strip()
    raw_sequences = re.split(r'\n\n', raw_text)

    text = []
    labels = []
    for raw_seq in raw_sequences:
        tmp_sent = []
        tmp_tags = []
        for raw_line in raw_seq.split('\n'):
            splits = raw_line.split(' ')
            tmp_word = ''.join([i if ord(i) < 128 else '-' for i in splits[0]])  #replace non ASCII char
            tmp_sent.append(tmp_word)
            tmp_tag = splits[-1]
            tmp_tags.append(tmp_tag)

        text.append(tmp_sent)
        labels.append(tmp_tags)

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

def preprocess_entities(entities, ent_aggr, ent_del_l):
    aggregations = dict()
    for k, v in ent_aggr.items():
        aggregations.update({'B-' + k: 'B-' + v,
                             'I-' + k: 'I-' + v})

    for ent in ent_del_l:
        aggregations.update({'B-' + ent: 'O',
                             'I-' + ent: 'O'})

    new_entities = [[aggregations.get(e, e) for e in seq] for seq in entities]

    return new_entities

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
