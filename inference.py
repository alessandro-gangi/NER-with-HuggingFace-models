import argparse
import os
from datetime import datetime
from os import path
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, TokenClassificationPipeline
from config import MODELS_DIR, DATASETS_DIR, SPECIAL_TOKENS
from utils.generic_utils import uniquify_filename

# Command line parameters
parser = argparse.ArgumentParser(description='NER with HuggingFace models')
parser.add_argument('model', type=str,
                    help='Name of a specific model previously saved inside "models" folder'
                         ' or name of an HuggingFace model')
parser.add_argument('doc', type=str,
                    help='Name of a specific .txt document present inside "datasets" folder. The document '
                         'should contain only plain text (without labels). Sequences should be separated by \n')
parser.add_argument('-noscores', action='store_true', help='If set, prediction scores wont be saved in output file')


def read_document(filepath):
    """
    Read a given document and return its content as a list of sequences
    :param filepath: str
        path to the document
    :return: list
        list of strings representing the content of the document
    """
    doc_sequences = []
    with open(filepath, encoding='utf-8') as fp:
        line = fp.readline()
        while line:
            doc_sequences.append(line)
            line = fp.readline()

    return doc_sequences


def write_inference_result(result: list, output_dir: str, output_filename: str, include_scores=False):
    """
    Write inference results on file (text with predictions).
    :param result: list
        list  of lists of tuples containing words, predicted labels and associated scores
    :param output_dir: str
        path to folder where result will be saved
    :param output_filename: str
        name of the results file that will be saved
    :param include_scores: bool
        whether to include scores in results file or not
    :return: void
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filepath = uniquify_filename(path.join(output_dir, output_filename))
    with open(output_filepath, "w", encoding='utf-8') as writer:
        for seq_with_preds in result:
            for word_with_pred in seq_with_preds:
                if include_scores:
                    word, tag, score = word_with_pred[0], word_with_pred[1], word_with_pred[2]
                    writer.write('%s %s %s\n' % (word, tag, score))
                else:
                    word, tag = word_with_pred[0], word_with_pred[1]
                    writer.write('%s %s\n' % (word, tag))
            writer.write('\n')


if __name__ == '__main__':
    args = parser.parse_args()
    model_name_or_path = args.model
    is_a_presaved_model = len(args.model.split('_')) > 1

    # Load tokenizer
    tokenizer_name_or_path = path.join(MODELS_DIR, model_name_or_path) if is_a_presaved_model else model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_name_or_path)

    # Load a specific, previously fine-tuned model or use one of the HuggingFace models
    model_name_or_path = path.join(MODELS_DIR, model_name_or_path) if is_a_presaved_model else model_name_or_path
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, return_dict=True)

    nlp = TokenClassificationPipeline(task='ner',
                                      framework='pt',
                                      model=model,
                                      tokenizer=tokenizer,
                                      grouped_entities=False,
                                      device=torch.cuda.current_device() if torch.cuda.is_available() else -1,
                                      ignore_labels=[])

    document = read_document(path.join(DATASETS_DIR, args.doc))
    predictions = nlp(document)

    # Generate a new document (as list of sequences) with the labels predicted by the model
    document_with_preds = []
    for seq_pred in predictions:
        seq_with_preds = []

        for token_pred in seq_pred:
            word, entity, score = token_pred['word'], token_pred['entity'], round(token_pred['score'], 3)
            if word in SPECIAL_TOKENS:
                # If word is a special token, we just skip it
                pass

            elif word.startswith('##'):
                # A word starting with '##' means that it's a portion of the previous
                # word so we concatenate the '##' word with the previous one
                seq_with_preds[-1][0] += word[2:]

            else:
                # Base case: add a tuple containing word, tag and score
                seq_with_preds.append([word, entity, score] if not args.noscores else [word, entity])

        document_with_preds.append(seq_with_preds)

    # Writing inference result
    today_date_str = datetime.now().strftime("%Y%m%d")
    model_infer_dir = path.join(*[model_name_or_path, 'inferences', today_date_str + '_' + args.doc.split('.', 1)[0]])
    write_inference_result(document_with_preds, model_infer_dir, 'results.txt', include_scores=not args.noscores)
    print(f"Inference result saved inside {model_infer_dir}.")
