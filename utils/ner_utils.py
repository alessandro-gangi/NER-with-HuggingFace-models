import json
from spacy.lang.it import Italian
from spacy.gold import biluo_tags_from_offsets
import re
from pathlib import Path
#import pandas as pd
#from pandas._libs import json
from transformers import TrainingArguments


def read_dataset(path, data_format='doccano'):
    """
    """
    if data_format == 'doccano':
        return read_doccano_dataset(path)
    elif data_format == 'conll':
        return read_conll_dataset(path)


def read_doccano_dataset(path):
    file_path = Path(path)
    raw_text = file_path.read_text(encoding='utf8').strip()

    json_strings = raw_text.split('\n')

    texts_ids, texts, labels = [], [], []
    for json_str in json_strings:
        json_obj = json.loads(json_str)

        texts_ids.append(json_obj['id'])
        texts.append(json_obj['text'])
        labels.append([(lab[0], lab[1], lab[2]) for lab in json_obj['labels']])

    # Spacy gold tokenizer to tokenize text and tags tokens with provided tags
    nlp = Italian()
    output_texts_full = []
    output_tags_full = []
    output_texts_reduced = []
    output_tags_reduced = []

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

        if '-' not in iob_tags:
            output_texts_reduced.append(tokenized_text)
            output_tags_reduced.append(iob_tags)
        output_texts_full.append(tokenized_text)
        output_tags_full.append(iob_tags)

    return output_texts_reduced, output_tags_reduced


def read_conll_dataset(path):
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
