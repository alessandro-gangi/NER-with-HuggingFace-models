import numpy as np
import torch.tensor
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerFast


def encode_labels(labels, label2id, encoded_text):
    """
    Encode labels according to how text was encoded
    :param labels: list
        list o labels associated with the (non encoded) text
    :param label2id: dict
        dictionary to retrieve labels ids given labels
    :param encoded_text: list
        encoded text useful to get the offset mapping (used to encode labels properly)
    :return: list
        encoded_labels
    """

    labels_ids = [[label2id[tag] for tag in sent] for sent in labels]
    encoded_labels = []

    for i, (sent_labels_ids, sent_offset) in enumerate(zip(labels_ids, encoded_text.offset_mapping)):
        # create an empty array of -100
        sent_enc_labels = np.ones(len(sent_offset), dtype=int) * -100
        arr_offset = np.array(sent_offset)

        # set labels whose first offset position is 0 and the second is not 0
        # print(f"len of sent_labels: {len(sent_labels)}")
        try:
            sent_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = sent_labels_ids
            encoded_labels.append(sent_enc_labels.tolist())
        except ValueError as err:
            print(f"encode_labels ERROR: {err}.\nUsually this error is solved by "
                  f"increasing max_sent_length or setting it to None.\nCHECK SENTENCES n{i}")
            raise ValueError

    return encoded_labels


class CustomNERDataset(Dataset):
    def __init__(self,
                 text,
                 labels,
                 label2id: dict,
                 tokenizer: PreTrainedTokenizerFast,
                 max_seq_length: int = None,  # Max sentence length to apply padding/truncation
                 tk_padding=True,  # Tokenizer applies padding to have standard length sentences
                 tk_truncation=True,  # Tokenizer applies truncation to have standard length sentences
                 ):
        self.encoded_text = tokenizer(text,
                                      max_length=max_seq_length,
                                      is_pretokenized=True,
                                      return_offsets_mapping=True,
                                      padding=tk_padding,
                                      truncation=tk_truncation)
        self.encoded_labels = encode_labels(labels, label2id, self.encoded_text)
        self.encoded_text.pop("offset_mapping")

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoded_text.items()}
        item['labels'] = torch.tensor(self.encoded_labels[idx])
        return item

    def __len__(self):
        return len(self.encoded_labels)
