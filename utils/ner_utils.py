import json
import warnings
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.it import Italian
from spacy.gold import biluo_tags_from_offsets


def read_data(path, prep_entities=None, split=(0.8, 0.2), seed=42):
    # Read data and build dataframe
    raw_data = Path(path).read_text(encoding='utf8').strip()
    jsons_data = raw_data.split('\n')
    keys_needed = ['text', 'labels']
    dict_data = [{k: v for k, v in json.loads(el).items() if k in keys_needed}
                 for el in jsons_data]
    df = pd.DataFrame(dict_data)

    # Preprocess data
    df = preprocess_data(df, prep_entities=prep_entities)

    # Split data in training and test set
    train_df, test_df = split_data(df, split, seed=seed)

    # Extract text, labels and indexes (to replicate the same split in future)
    train_texts, train_labels, train_indexes = train_df['text'].to_list(), train_df['labels'].to_list(), train_df.index.to_series()
    test_texts, test_labels, test_indexes = test_df['text'].to_list(), test_df['labels'].to_list(), test_df.index.to_series()

    return train_texts, test_texts, train_labels, test_labels, train_indexes, test_indexes


def split_data(data_df, split, seed=42):
    def least_frequent(alist):
        return min(set(alist), key=alist.count) if alist else 'O'

    def most_frequent(alist):
        return max(set(alist), key=alist.count) if alist else 'O'

    # 3 different splits are tried: tue first two are stratified, the last is
    # juts a shuffle split
    labels = data_df['labels']
    strat_functions = [least_frequent, most_frequent, None]

    train_df, test_df = None, None
    for i, f in enumerate(strat_functions):
        y_strat = [f(l) for l in [list(filter(lambda x: x != 'O', sublist))
                                  for sublist in labels if sublist != 'O']] if f else None

        try:
            train_df, test_df = train_test_split(data_df, test_size=split[1],
                                                 stratify=y_strat, random_state=seed,
                                                 shuffle=True)
            print(f'DATA SPLIT type: {i}\t(0:least_frequent-strat, 1:most_frequent-strat, 2:no-stratified)')
            break
        except ValueError:
            continue

    return train_df, test_df


def preprocess_data(data_df, prep_entities):
    nlp = Italian()
    for index, row in data_df.iterrows():
        text = row['text']
        labels = row['labels']

        # fixing starting/ending space misalignments
        labels = fix_labels_misalignments(text=text, labels=labels)
        if labels is None:  # unable to fix -> skip this text
            data_df.drop(index, inplace=True)
            data_df.reset_index(drop=True)
            continue

        # tokenize text and labels
        doc = nlp(text)
        text = [token.text for token in doc]
        warnings.filterwarnings("ignore", message=r"\[W030\]", category=UserWarning)
        labels = biluo_tags_from_offsets(doc, labels)

        # fix tokenization
        text, labels = fix_tokenization(tok_text=text, labels=labels)

        # apply entities aggregations and deletion
        aggregations, to_filter = prep_entities
        labels = filter_and_aggregate(labels=labels, aggregations=aggregations,
                                      to_filter=to_filter)
        assert len(text) == len(labels)

        # update dataframe
        data_df.at[index, 'text'] = text
        data_df.at[index, 'labels'] = labels

    return data_df


def filter_and_aggregate(labels, aggregations: dict, to_filter: list):
    labels = [l if (l != 'o' and l[2:] not in to_filter) else 'O' for l in labels]
    labels = [l[:2] + aggregations[l[2:]] if l[2:] in aggregations.keys() else l for l in labels]

    return labels


def fix_tokenization(tok_text, labels):
    # remove empty tokens and their labels
    for i, token in enumerate(tok_text):
        if token.isspace():
            tok_text.pop(i)
            labels.pop(i)

    # convert labels to IOB standard
    labels = [l.replace('U-', 'B-') for l in labels]
    labels = [l.replace('L-', 'I-') for l in labels]

    # Replace remained misaligned tag '-' with 'O' tag
    labels = [l if l != '-' else 'O' for l in labels]

    return tok_text, labels


def fix_labels_misalignments(text, labels):
    try:
        # fixing ending space misalignment
        labels = [(lab[0], lab[1], lab[2]) if ' ' != text[lab[1] - 1]
                  else (lab[0], lab[1] - 1, lab[2]) for lab in labels]
        # fixing starting space misalignment
        labels = [(lab[0], lab[1], lab[2]) if ' ' != text[lab[0]]
                  else (lab[0] + 1, lab[1], lab[2]) for lab in labels]

    except IndexError:
        print(f"-----------------------\n"
              f"There is a problem with the following document in the corpus: please check its "
              f"annotations indexes.(this document will be skipped for now)\n"
              f"Text: {text}\nLabels: {labels}\n"
              f"-----------------------")
        return None

    return labels
