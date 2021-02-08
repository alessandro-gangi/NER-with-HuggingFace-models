import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict

from langdetect.lang_detect_exception import LangDetectException
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from spacy.lang.it import Italian
from spacy.lang.en import English
from spacy.gold import biluo_tags_from_offsets
from langdetect import detect


def read_data(path: str, prep_entities=None, split=(0.8, 0.2), seed=None):
    """
    Read the dataset and split it into training and evaluation sets
    :param path: path of the dataset
    :param prep_entities: tuple of len 2
        - tuple[0] is a dictionary of aggregations ('key' entity will be mapped to 'value' entity)
        - tuple[1] is a list of entities not to be considered in training/evaluation
    :param split: training and evaluation sizes
    :param seed: seed for reproducibility
    :return: train_texts, test_texts, train_labels, test_labels, train_indexes, test_indexes
    """
    # Read data and build dataframe
    raw_data = Path(path).read_text(encoding='utf8').strip()
    jsons_data = raw_data.split('\n')
    keys_needed = ['id', 'text', 'labels', 'meta']
    dict_data = [{k: v for k, v in json.loads(el).items() if k in keys_needed}
                 for el in jsons_data]
    df = pd.DataFrame(dict_data)

    # Get the document localyzer dictionary
    df, dfidx2corpus_idx = get_doc_localyser_dict(df)

    # Preprocess data
    if prep_entities:
        df = preprocess_data(df, prep_entities=prep_entities)

    # Split data in training and test set
    train_df, test_df = split_data(df, split, seed=seed)

    # Extract text, labels and indexes (to replicate the same split in future)
    train_texts, train_labels, train_indexes = train_df['text'].to_list(), train_df[
        'labels'].to_list(), train_df.index.to_series()
    test_texts, test_labels, test_indexes = test_df['text'].to_list(), test_df[
        'labels'].to_list(), test_df.index.to_series()

    return train_texts, test_texts, train_labels, test_labels, train_indexes, test_indexes, dfidx2corpus_idx


def get_doc_localyser_dict(data_df):
    """
    Build a dictionary with key=index (of data_df rows) and value = (corpus_name, doc_index_inside_the_corpus)
    :param data_df: pd.Dataframe
        dataframe with columns 'id', 'text', 'labels' and 'meta' representing the whole dataset
    :return: the same dataframe 'data_df' and the dictionary just created
    """

    # Add the new column 'corpus' containing corpora names associated to the row
    # and sort the dataframe based on Doccano IDs
    data_df['corpus'] = pd.Series(data_df['meta'][i]['corpus_name'] for i in range(data_df['meta'].size))
    df = data_df.sort_values('id')

    # Create a tmp dictionary and the one to return as result
    corpus2lastidx = {corpus: 1 for corpus in set(df['corpus'])}
    idx2corpus_doc = {}  # to return

    # Iterate over the rows and populate the result dictionary:
    # key=dataframe index of the row, value=(name of the corpus, index of document inside the corpus)
    for i in range(len(df.index)):
        c = df.iloc[i, 4]
        idx2corpus_doc[df.index[i]] = (c, corpus2lastidx[c])
        corpus2lastidx[c] += 1

    # Drop the column 'corpus' we added before and sort (back) the dataframe based on dataframe indexes
    df = df.drop(columns=['corpus', ])
    df.sort_index(inplace=True)

    return df, idx2corpus_doc


def split_data(data_df, split: tuple, seed=None):
    """
    Split the dataframe into training and test/evaluation
    :param data_df: dataframe containing the whole dataset
    :param split: tuple of len 2 with training and test/evaluation sizes
    :param seed: seed for reproducibility of the split
    :return: train_df, test_df
    """

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
        y_strat = [f([lab[2:] for lab in l]) for l in [list(filter(lambda x: x != 'O', sublist))
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
    """
    Preprocess the input dataframe by:
    - fixing misalignments problems
    - tokenizing texts and converting labels to IOB format
    - applying entities aggregations and deletion
    :param data_df: input datframe to be preprocessed
    :param prep_entities: tuple of len 2
        - tuple[0] is a dictionary of aggregations ('key' entity will be mapped to 'value' entity)
        - tuple[1] is a list of entities not to be considered in training/evaluation
    :return: preprocessed dataframe
    """
    nlp_it, nlp_en = Italian(), English()
    for index, row in data_df.iterrows():
        id = row['id']
        text = row['text']
        labels = row['labels']

        # fixing starting/ending space misalignments
        labels = fix_labels_misalignments(id=id, text=text, labels=labels)
        if labels is None:  # unable to fix -> skip this text
            data_df.drop(index, inplace=True)
            data_df.reset_index(drop=True)
            continue

        # get language and tokenize text and labels
        try:
            language = detect(text)
        except LangDetectException:
            language = 'it'
        doc = nlp_en(text) if language == 'en' else nlp_it(text)

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
    """
    Process labels by removing (converting in 'O') some entities and by aggregating others
    :param labels: list of labels to be processed
    :param aggregations: dictionary where 'key' entity has to be aggregated to 'value' entity
    :param to_filter: list of entities to be removed
    :return: processed labels
    """
    labels = [l if (l != 'O' and l[2:] not in to_filter) else 'O' for l in labels]
    labels = [l[:2] + aggregations[l[2:]] if l[2:] in aggregations.keys() else l for l in labels]

    return labels


def fix_tokenization(tok_text, labels):
    """
    Fix tokenization problems and convert labels (from Spacy) to IOB format
    :param tok_text: list of tokens (a single document)
    :param labels: list of labels (labels associated to the document)
    :return: fixed (tok_text, labels)
    """
    # remove empty tokens and their labels
    for i, token in enumerate(tok_text):
        if token.isspace():
            tok_text.pop(i)
            labels.pop(i)

    # convert labels to IOB standard
    labels = [l.replace('U-', 'B-') for l in labels]
    labels = [l.replace('L-', 'I-') for l in labels]

    # Replace remained misaligned tags '-' with 'O' tags
    labels = [l if l != '-' else 'O' for l in labels]

    return tok_text, labels


def fix_labels_misalignments(id, text, labels):
    """
    Fix annotations misalignments caused by an empty space at the beginning of the annotation span or at its end
    :param id: document id
    :param text: document (list of tokens) text
    :param labels: document (list of) labels
    :return:
    """
    try:
        # fixing ending space misalignment
        labels = [(lab[0], lab[1], lab[2]) if ' ' != text[lab[1] - 1]
                  else (lab[0], lab[1] - 1, lab[2]) for lab in labels]
        # fixing starting space misalignment
        labels = [(lab[0], lab[1], lab[2]) if ' ' != text[lab[0]]
                  else (lab[0] + 1, lab[1], lab[2]) for lab in labels]

    except IndexError:
        print(f"\nThere is a problem with the following document in the corpus: please check its "
              f"annotations indexes.(this document will be skipped for now)\n"
              f"ID: {id}\nText: {text}\nLabels: {labels}\n"
              f"-----------------------")
        return None

    return labels


def process_predictions(predict_results):
    """
    Process prediction results of the model
    :param predict_results: tuple of len 2
        - tuple[0] represents predictions made by the mode
        - tuple[1] represents (raw/unprocessed) true label_ids
    :return: processed (flattened) predicted label_ids and true label_ids
    """
    predictions, true_label_ids_raw = predict_results.predictions, predict_results.label_ids
    batch_size, seq_len = true_label_ids_raw.shape
    # print(f'Batch size: {batch_size}, Seq len: {seq_len}')

    # predictions[i, j] is the logits vector of the j-th token of the i-th document
    # so we need to take the argmax to turn this vector into the corresponding label_id
    preds_label_ids_raw = np.argmax(predictions, axis=2)
    # print(f'Shape true_raw: {true_label_ids_raw.shape} Shape pred_raw: {preds_label_ids_raw.shape}')

    # true_label_ids_raw[i, j] is the label_id of the j-th token of the i-th document
    # if this value == -100 then it's a special token or a padding token and we don't want to consider it
    # *notice how, after filtering out these values, each row will have different length
    true_label_ids = [[] for _ in range(batch_size)]
    preds_label_ids = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if true_label_ids_raw[i, j] != -100:
                true_label_ids[i].append(true_label_ids_raw[i][j])
                preds_label_ids[i].append(preds_label_ids_raw[i][j])
    # print(f'Shape true: ({len(true_label_ids)}, varying) Shape pred: ({len(preds_label_ids)}, varying)')

    # Now we need to flatten our two lists of lists into lists in order to compute metrics
    #
    # TEST: Now the method does not flatten the lists anymore (they're flattened in finetune.py)
    #
    # preds_label_ids_flat = [lab_id for doc_lab_ids in preds_label_ids for lab_id in doc_lab_ids]
    # true_label_ids_flat = [lab_id for doc_lab_ids in true_label_ids for lab_id in doc_lab_ids]
    # print(f'Shape true_flat: ({len(true_label_ids_flat)}, ) Shape pred_flat: ({len(preds_label_ids_flat)}, )')

    return preds_label_ids, true_label_ids


def get_metric_scores(preds_label_ids_flat, true_label_ids_flat, label_ids, labels):
    """
    compute, for each label, metric scores on predictions of the model
    :param preds_label_ids_flat: list of label_ids predicted by the model
    :param true_label_ids_flat: list of true label_ids
    :param label_ids: list of different label_ids used by the model
    :param labels: list of different labels (associated to label_ids) used by the model
    :return: dataframe containing metric scores associated to labels with columns: precision, recall, f1-score,
        support
    """
    # We need a list of labels (and corresponding ids) to compute metrics scores through classification report
    # * notice how we don't consider the 'O' label when computing scores
    label_ids.remove(labels.index('O'))
    labels.remove('O')
    scores = classification_report(true_label_ids_flat,
                                   preds_label_ids_flat,
                                   labels=label_ids,
                                   target_names=labels,
                                   output_dict=True)

    # Create a dataframe containing, for each label, their metrics scores
    metrics = ['precision', 'recall', 'f1-score', 'support']
    scores_data = []
    for label, score in scores.items():
        label_scores = [label]
        for metric in metrics:
            label_scores.append(round(score[metric], 3))
        scores_data.append(label_scores)
    df_scores = pd.DataFrame.from_records(scores_data, columns=['label'] + metrics)

    return df_scores


def get_predictions_errors(true_label_ids, preds_label_ids, eval_texts, eval_indexes, id2label, idx2corpus_idx):
    """
    Build a dataframe containing of errors with this structure: the first column (True) contains
    the true label, the second (Pred) contains the predicted label. The fourth (Errors) contains a list of tuple (of len 2) with
    the following structure:
    - t[0] is the index of the document containing the error
    - t[1] is the index of the token (inside the document t[0]) that was misclassified (it was predicted as 'Pred' but
    it was 'True'.
    The third column(Num) is the length of the list in column 4 and so it contains the number of total tokens that were
    classified as 'Pred' but were 'True'
    :param true_label_ids: list of lists of int
        True labels associated to each token of each document
    :param preds_label_ids: list of lists of int
        Predicted labels associated to each token of each document
    :param eval_texts: list of lists of str
        Evaluation texts data obtained by reading and splitting the whole dataset
    :param eval_indexes: pd.Series
        Series representing the indexes (based on the whole dataset) od the documents present in the eval dataset
    :param id2label: dict
        Dictionary with ley=label_id and value=label
    :param idx2corpus_idx: dict
        Dictionary with key=index and value=(origin_corpus_name, index_of_doc_inside_that_corpus)
    :return: pd.Dataframe
        Dataframe with 4 columns (True, Pred, Num, Errors)
    """
    eval_indexes_l = eval_indexes.to_list()
    errors = defaultdict(list)

    for i, true_doc_label_ids in enumerate(true_label_ids):
        for j, true_label_id in enumerate(true_doc_label_ids):
            if true_label_id != preds_label_ids[i][j]:
                true_label = id2label[true_label_id]
                preds_label = id2label[preds_label_ids[i][j]]
                corpus_name, doc_idx = idx2corpus_idx[eval_indexes_l[i]]
                token = eval_texts[i][j]
                errors[(true_label, preds_label)].append((corpus_name, doc_idx, "'" + token + "'"))

    errors_l = [(k[0], k[1], len(v), v) for k, v in errors.items()]

    df_errors = pd.DataFrame(errors_l, columns=['True', 'Pred', 'Num', 'Errors'])
    # df_errors = df_errors.sort_values(by=['Num'], ascending=False)

    return df_errors


def get_confusion_matrix(true_label_ids_flat, preds_label_ids_flat, labels, label_ids, normalize='true'):
    """
    Build confusion matrix based on predictions of the model
    :param true_label_ids_flat: list of predicted label_ids
    :param preds_label_ids_flat: list of trye label_ids
    :param labels: list of different labels used by the model
    :param label_ids: list of different label_ids used by the model
    :param normalize: string value to normalize matrix by rows ('true'), by columns ('pred') or both ('all')
    :return: dataframe representing the confusion matrix
    """

    cm = confusion_matrix(true_label_ids_flat, preds_label_ids_flat, labels=label_ids, normalize=normalize)
    cm_df = pd.DataFrame(cm, columns=labels)
    # round values
    for l in labels:
        cm_df[l] = cm_df[l].round(3)
    # add left column with labels
    cm_df.insert(loc=0, column='', value=labels)
    cmap = sns.light_palette("grey", as_cmap=True)
    cm_df = cm_df.style.background_gradient(cmap=cmap)

    return cm_df
