import argparse
import time
from datetime import datetime
from os import path
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig, Trainer, TrainingArguments, \
    EvalPrediction
from ner.custom_ner_dataset import CustomNERDataset
from utils.generic_utils import uniquify_filename
from utils.ner_utils import read_data, process_predictions, get_metric_scores, get_confusion_matrix
from utils.plot_utils import plot_results
from config import MODELS_DIR, DATASETS_DIR, ENTITIES_AGGREGATIONS, ENTITIES_TO_FILTER
from utils.results_utils import save_training_infos, save_evaluation_result

#
# Command line parameters
#

parser = argparse.ArgumentParser(description='NER with HuggingFace models')

# Mandatory
parser.add_argument('model', type=str,
                    help='Name of a specific model previously saved inside "models" folder'
                         ' or name of an HuggingFace model')
parser.add_argument('dataset', type=str,
                    help='Name of a specific Doccano dataset present inside "datasets" folder.')

# Optional
parser.add_argument('-tok', default='', type=str, help='Name of a specific tokenizer (check HuggingFace list). '
                                                       'If not provided, an automatic tokenizer will be used')
parser.add_argument('-config', default='', type=str, help='Name of a specific model configuration (check HuggingFace'
                                                          ' list). If not provided, an automatic configuration will'
                                                          ' be used')
parser.add_argument('-splitseed', default=None, type=int, help='Seed for reproducibility when sampling dataset. '
                                                               'Default is None (random seed).')
parser.add_argument('-noentprep', action='store_true', help='If not set, entities will be preprocessed according to '
                                                            'the config file')
parser.add_argument('-notrain', action='store_true', help='If set, training will be skipped')
parser.add_argument('-noeval', action='store_true', help='If set, evaluation phase will be skipped')
parser.add_argument('-plot', action='store_true', help='If set, charts will be plotted')
parser.add_argument('-maxseqlen', default=None, type=int,
                    help='Value used by tokenizer to apply padding or truncation to sequences. Default is '
                         'None=max_value=512')
parser.add_argument('-epochs', default=2, type=int, help='Number of epochs during training')
parser.add_argument('-warmsteps', default=500, type=int, help='Number of warm-up steps before training')
parser.add_argument('-wdecay', default=0.00, type=float, help='Weight decay to use during training')
parser.add_argument('-trainbatch', default=32, type=int, help='Per device batch size during training')
parser.add_argument('-evalbatch', default=64, type=int, help='Per device batch size during evaluation')
parser.add_argument('-logsteps', default=100, type=int, help='Number of training steps between 2 logs')

"""
def align_predictions(predictions: np.ndarray, true_labels_ids: np.ndarray, binarize=True):
    
    Support method to re-align the labels and convert them to a binary (class) representation
    :param predictions: np.ndarray
        predictions of the model. Use argmax on second axis to get the predicted ids
    :param true_labels_ids: np.ndarray
        true labels ids of the data
    :param binarize: bool
        whether to convert labels to binary format or not
    :return: re-aligned and binarized true labels and prediction labels
    

    preds_labels_ids = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds_labels_ids.shape

    true_labels = [[] for _ in range(batch_size)]
    preds_labels = [[] for _ in range(batch_size)]

    # Re-align the labels (they were unaligned to match the text encoded by tokenizer)
    for i in range(batch_size):
        for j in range(seq_len):
            if true_labels_ids[i, j] != -100:
                true_labels[i].append(model_config.id2label[true_labels_ids[i][j]])
                preds_labels[i].append(model_config.id2label[preds_labels_ids[i][j]])

    # Convert from multi-label representation to a binary one
    if binarize:
        classes = list(model_config.id2label.values())
        lb = LabelBinarizer()
        lb.classes_ = classes

        bin_preds_labels, bin_true_labels = [], []
        for i, seq_preds_labels in enumerate(preds_labels):
            bin_preds_labels.append(lb.transform(seq_preds_labels))
            bin_true_labels.append(lb.transform(true_labels[i]))

        preds_labels = bin_preds_labels
        true_labels = bin_true_labels

    return preds_labels, true_labels
"""
"""
def compute_metrics(p: EvalPrediction):
    
    Compute scores of model according to some metrics
    :param p: EvalPrediction
        object containing predictions and true labels ids
    :return: dict
        containing scores
    
    preds_labels, true_labels = align_predictions(p.predictions, p.label_ids)
    flat_preds_labels = [item for sublist in preds_labels for item in sublist]
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    labels = list(model_config.id2label.keys())
    target_names = list(model_config.id2label.values())

    # Do not consider 'O' label in evaluation
    labels.remove(target_names.index('O'))
    target_names.remove('O')

    # Get scores
    scores = classification_report(flat_true_labels,
                                   flat_preds_labels,
                                   labels=labels,
                                   target_names=target_names,
                                   output_dict=True)

    # Covert scores into dictionary with scalar values
    # *Trainer (tensorboard) needs this format
    scalar_scores = dict()
    for key, value in scores.items():
        for sub_key, sub_value in value.items():
            new_key = key + '_' + sub_key
            scalar_scores.update({new_key: sub_value})

    return scalar_scores
"""

if __name__ == '__main__':
    args = parser.parse_args()
    today_date_str = datetime.now().strftime("%Y%m%d")

    # Setting up directories according to the model_name provided in command line
    model_name_or_path = args.model
    is_a_presaved_model = len(model_name_or_path.split('_')) > 1

    model_output_dir = path.join(MODELS_DIR, model_name_or_path + ('_' + today_date_str if not args.notrain else ''))
    if not args.notrain:
        model_output_dir = uniquify_filename(model_output_dir)

    model_cache_dir = path.join('cache', args.model)

    # If we are only evaluating a model then we save the results (and the logs) inside a specific folder
    model_eval_dir = uniquify_filename(path.join(*[model_output_dir, 'evaluations', today_date_str + '_'
                                                   + args.evalset.split('.')[0]]))

    model_logs_dir = path.join(model_eval_dir if args.notrain else model_output_dir, 'logs')

    print(f"Is a pre-saved model? {is_a_presaved_model}")
    print(f"Model Output dir: {model_output_dir}")
    print(f"Model Cache dir: {model_cache_dir}")
    print(f"Model Eval dir: {model_eval_dir}")
    print(f"Model Logs dir: {model_logs_dir}")

    train_texts, eval_texts, \
    train_labels, eval_labels, \
    train_indexes, eval_indexes = read_data(path=path.join(DATASETS_DIR, args.dataset),
                                            prep_entities=(ENTITIES_AGGREGATIONS,
                                                           ENTITIES_TO_FILTER),
                                            split=(0.8, 0.2),
                                            seed=args.splitseed)

    # Load a specific model configuration or automatically use the one associated to the model
    config_name_or_path = args.config if args.config \
        else path.join(MODELS_DIR, model_name_or_path) if is_a_presaved_model else model_name_or_path
    print(f"Config name: {config_name_or_path}")
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=config_name_or_path,
        cache_dir=path.join(model_cache_dir, 'config')
    )

    # If we are training the model we have to overwrite the configuration parameters related to labels
    # checking the specified train set. If we are evaluating a new HuggingFace model, we do the same thing
    # (we have to tell the model which labels to use).
    # Else (we are evaluating a previously saved model) we use the configuration parameters.
    unique_labels = set()
    if not args.notrain or not is_a_presaved_model:
        unique_labels = set()
        unique_labels.update([l for seq in (train_labels if not args.notrain else []) for l in seq])
        unique_labels.update([l for seq in (eval_labels if not args.noeval else []) for l in seq])
        num_labels = len(unique_labels)
        label2id = {lab: lab_id for lab_id, lab in enumerate(unique_labels)}
        id2label = {lab_id: lab for lab, lab_id in label2id.items()}

        model_config.num_labels = num_labels
        model_config.id2label = id2label
        model_config.label2id = label2id

    # Load a specific tokenizer or automatically use the one associated to the model
    tokenizer_name_or_path = args.tok if args.tok \
        else path.join(MODELS_DIR, model_name_or_path) if is_a_presaved_model else model_name_or_path
    print(f"Tokenizer name: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name_or_path,
        cache_dir=path.join(model_cache_dir, 'tokenizer'),
        use_fast=True
    )

    # Load a specific, previously fine-tuned model or use one of the HuggingFace models
    model_name_or_path = path.join(MODELS_DIR, model_name_or_path) if is_a_presaved_model else model_name_or_path
    print(f"Model name: {model_name_or_path}")
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=model_config,
        from_tf=bool('.ckpt' in model_output_dir),
        cache_dir=path.join(model_cache_dir, 'model')
    )

    # Create train and eval dataset with texts and labels previously read
    train_dataset = CustomNERDataset(text=train_texts,
                                     labels=train_labels,
                                     label2id=model_config.label2id,
                                     tokenizer=tokenizer,
                                     max_seq_length=args.maxseqlen,
                                     tk_padding=True,
                                     tk_truncation=True)

    eval_dataset = CustomNERDataset(text=eval_texts,
                                    labels=eval_labels,
                                    label2id=model_config.label2id,
                                    tokenizer=tokenizer,
                                    max_seq_length=args.maxseqlen,
                                    tk_padding=True,
                                    tk_truncation=True) if not args.noeval else None

    # Customize training and evaluation
    training_arguments = TrainingArguments(output_dir=model_output_dir,
                                           num_train_epochs=args.epochs,
                                           warmup_steps=args.warmsteps,
                                           weight_decay=args.wdecay,
                                           per_device_train_batch_size=args.trainbatch,
                                           per_device_eval_batch_size=args.evalbatch,
                                           logging_dir=model_logs_dir,
                                           logging_steps=args.logsteps)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset
    )

    # Fine-tune model and save it together with its tokenizer
    if not args.notrain:
        print(f"Training {model_name_or_path} now...")
        train_start_time = time.time()
        trainer.train()
        train_elapsed_time = (time.time() - train_start_time)

        trainer.save_model(output_dir=model_output_dir)
        tokenizer.save_pretrained(save_directory=model_output_dir)
        if trainer.is_world_process_zero():
            save_training_infos(model_name=model_name_or_path,
                                train_dataset_name=args.trainset,
                                num_epochs=args.epochs,
                                duration=train_elapsed_time,
                                output_dir=model_output_dir,
                                output_filename='training_infos.txt')

        print(f"Model trained and saved inside {model_output_dir}.")
        print(f"Tokenizer saved inside {model_output_dir}.")

    # Evaluate model, write a results file (saved together with the model) and plot graphs
    if not args.noeval:
        eval_start_time = time.time()
        predict_results = trainer.predict(eval_dataset)
        eval_elapsed_time = (time.time() - eval_start_time)

        preds_label_ids_flat, true_label_ids_flat = process_predictions(predict_results)
        label_ids = list(model_config.id2label.keys())
        labels = list(model_config.id2label.values())

        # Compute scores (precision, recall, f1-score, support) dataframe
        df_scores = get_metric_scores(preds_label_ids_flat, true_label_ids_flat, label_ids, labels)

        # Compute confusion matrix dataframe
        df_confmatrix = get_confusion_matrix(true_label_ids_flat, preds_label_ids_flat, labels, label_ids,
                                             normalize='true')

        # Write file
        if trainer.is_world_process_zero():
            save_evaluation_result(df_scores=df_scores,
                                   df_conf_matrix=df_confmatrix,
                                   model_name=model_name_or_path,
                                   eval_dataset_name=path.join(DATASETS_DIR, args.evalset),
                                   split_indexes=(train_indexes, eval_indexes),
                                   duration=eval_elapsed_time,
                                   output_dir=model_eval_dir,
                                   output_filename='eval_results.xlsx',
                                   aggregations=ENTITIES_AGGREGATIONS,
                                   deleted_entities=ENTITIES_TO_FILTER)

        # Plot charts #TODO: fix plots (now 'scores' is a dataframe)
        #if args.plot:
        #    plot_results(eval_scores, model_eval_dir)

        print(f"{model_name_or_path} evaluated: eval_results saved in {model_eval_dir}.")
