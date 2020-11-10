import re
from operator import itemgetter
from pathlib import Path

"""

THIS FILE IS NOT USED BY finetune OR inference SCRIPT 

THESE UTILS WERE USED FOR CONVERTING DATASETS IN 
OTHER FORMATS OR TO EXTRACT SENTENCES WITH MANY ENTITIES INSIDE

"""


def main_method():
    # copy this in real -main- file
    """
    result = extract_rich_data(data_args.data_dir + 'prova.txt', tag_style="bio")
    sents = list()
    for s in result[:40]:
        sents.append([s[0]])

    with open(os.path.join(training_args.output_dir, 'selected_sents_conll.csv'), 'w', newline='',
              encoding='utf8') as out:
        writer = csv.writer(out)
        writer.writerows(sents)
    """


def extract_rich_data(path, tag_style="bio"):
    """
    FOR TEST ONLY
    """
    text, labels = read_dataset_conll(path)
    flat_text = [' '.join(seq) for seq in text]
    flat_labels = [' '.join(lab_seq) for lab_seq in labels]
    seq_scores = [0] * len(flat_text)

    # compute scores
    if tag_style == "bio":
        for i, seq in enumerate(flat_text):
            labels_found = set()
            for j, word in enumerate(text[i]):
                lab = labels[i][j]
                if 'B-' in lab:
                    if lab not in labels_found:
                        seq_scores[i] += 2
                        labels_found.add(lab)
                    else:
                        seq_scores[i] -= 1

            #if len(seq.split()) < 30:
             #   seq_scores[i] = 0
    else: #"IO"
        for i, seq in enumerate(flat_text):
            labels_found = set()
            last_label_found = ""
            for j, word in enumerate(text[i]):
                lab = labels[i][j]
                if 'I-' in lab:
                    if lab not in labels_found:
                        seq_scores[i] += 2
                        labels_found.add(lab)
                    elif lab != last_label_found:
                        seq_scores[i] += 1
                    last_label_found = lab

    result = sorted([list(x) for x in zip(flat_text, flat_labels, seq_scores)], key=itemgetter(2))[::-1]

    return result


def read_dataset_conll(path):
    """
    TEST ONLY: legge il dataset e restituisce frasi e labels
    - text: lista di liste di stringhe (parole)
    - labels: lista di liste di label
    """
    file_path = Path(path)
    raw_text = file_path.read_text().strip()
    raw_sequences = re.split(r'\n\t?\n', raw_text)
    text = []
    labels = []
    for raw_seq in raw_sequences:
        tmp_sent = []
        tmp_tags = []
        for raw_line in raw_seq.split('\n'):
            splits = raw_line.rsplit(sep=' ', maxsplit=4)
            tmp_tag = splits[-1]
            tmp_word = splits[0]
            tmp_sent.append(tmp_word)
            tmp_tags.append(tmp_tag)
        text.append(tmp_sent)
        labels.append(tmp_tags)

    return text, labels


def read_dataset_gmb(path):
    """
    TEST ONLY: legge il dataset e restituisce frasi e labels
    - text: lista di liste di stringhe (parole)
    - labels: lista di liste di label
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf8").strip()

    raw_lines = re.split(r'\n', raw_text)
    lines = list()
    for line in raw_lines:
        if line[0] == ',':
            lines.append(line[1:])
        else:
            lines.append(line)

    text = []
    labels = []

    tmp_sent_words = []
    tmp_sent_labels = []
    for line in lines:
        if line.startswith('Sentence: '):
            if tmp_sent_labels and tmp_sent_words:
                text.append(tmp_sent_words)
                labels.append(tmp_sent_labels)
            tmp_sent_words = []
            tmp_sent_labels = []
            _, word, pos, ne_label = line.split(',')
            tmp_sent_words.append(word)
            tmp_sent_labels.append(ne_label)

        else:
            try:
                splits = line.rsplit(sep=',', maxsplit=4)
                ne_label = splits[-1]
                word = splits[-3]
                tmp_sent_words.append(word)
                tmp_sent_labels.append(ne_label)
            except:
                pass

    return text, labels


def read_dataset_wikigold(path):
    """
    TEST ONLY: legge il dataset e restituisce frasi e labels
    - text: lista di liste di stringhe (parole)
    - labels: lista di liste di label
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf8").strip()
    raw_sequences = re.split(r'\n\n', raw_text)
    text = []
    labels = []
    for raw_seq in raw_sequences:
        tmp_sent = []
        tmp_tags = []
        for raw_line in raw_seq.split('\n'):
            tmp_word, tmp_tag = raw_line.split(' ')
            tmp_sent.append(tmp_word)
            tmp_tags.append(tmp_tag)
        text.append(tmp_sent)
        labels.append(tmp_tags)

    return text, labels


def read_dataset_wnut(path):
    """
    TEST ONLY: legge il dataset e restituisce frasi e labels
    - text: lista di liste di stringhe (parole)
    - labels: lista di liste di label
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf8").strip()
    raw_sequences = re.split(r'\n\t?\n', raw_text)
    text = []
    labels = []

    for raw_seq in raw_sequences:
        tmp_sent = []
        tmp_tags = []
        for raw_line in raw_seq.split('\n'):
            try:
                tmp_word, tmp_tag = raw_line.split('\t')
                tmp_sent.append(tmp_word)
                tmp_tags.append(tmp_tag)
            except:
                pass
        text.append(tmp_sent)
        labels.append(tmp_tags)

    return text, labels


def read_dataset_secfiling(path):
    """
    TEST ONLY: legge il dataset e restituisce frasi e labels
    - text: lista di liste di stringhe (parole)
    - labels: lista di liste di label
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf8").strip()
    raw_sequences = re.split(r'\n\t?\n', raw_text)
    text = []
    labels = []

    for raw_seq in raw_sequences:
        tmp_sent = []
        tmp_tags = []
        for raw_line in raw_seq.split('\n'):
            try:
                splits = raw_line.rsplit(sep=' ')
                tmp_word = splits[-4]
                tmp_tag = splits[-1]
                tmp_sent.append(tmp_word)
                tmp_tags.append(tmp_tag)
            except:
                print(raw_line)
                pass
        text.append(tmp_sent)
        labels.append(tmp_tags)

    return text, labels