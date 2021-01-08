import os
from datetime import timedelta
from os import path
import pandas as pd

def save_training_infos(model_name: str, train_dataset_name: str, num_epochs: int,
                        duration: float, output_dir: str, output_filename: str):
    """
    Write training infos on file.
    :param model_name: str
        Name of the model evaluated
    :param train_dataset_name: str
        name of the dataset used for training
    :param num_epochs: int
        number of training epochs
    :param duration: float
        training duration time
    :param output_dir: str
        path to folder where result will be saved
    :param output_filename: str
        name of the results file that will be saved
    :return: void
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filepath = path.join(output_dir, output_filename)
    with open(output_filepath, "w") as writer:
        writer.write('Trained model name: ' + model_name + '\n')
        writer.write('Dataset used for training: ' + train_dataset_name + '\n')
        writer.write('Num. epochs: ' + str(num_epochs) + '\n')
        writer.write('Training duration: ' + str(timedelta(seconds=duration)))


def save_evaluation_result(scores: dict, model_name: str, eval_dataset_name: str, labels: set,
                           duration: float, output_dir: str, output_filename: str):
    """
    Write metric results on file.
    :param scores: dict
        dictionary containing metric scores
    :param model_name: str
        Name of the model evaluated
    :param eval_dataset_name: str
        name of the dataset used for evaluation
    :param labels: set
        set of labels used during training/evaluation
    :param duration: float
        training duration time
    :param output_dir: str
        path to folder where result will be saved
    :param output_filename: str
        name of the results file that will be saved
    :return: void
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_metrics = set()  # support, recall, ecc
    unique_types = set()  # macro avg, micro avg, ecc
    lab_met2score = dict()
    type_met2score = dict()
    bylabel_data = []  # lab, sup, rec, prec, f1
    general_data = []  # type, sup, rec, prec, f1
    other_data = []  # key, value

    # Fill lab_met2score and type_met2score support dictionaries
    # or fil other data
    for k, score in scores.items():
        splits = k.split('_')
        if len(splits) == 3:
            _, lab_or_type, met = splits[0], splits[1], splits[2]
            unique_metrics.add(met)
            if lab_or_type not in labels:  # so it's a type
                unique_types.add(lab_or_type)
                type_met2score[(lab_or_type, met)] = score
            else:
                lab_met2score[(lab_or_type, met)] = score
        else:  # other (epoch, total_floss, eval_loss)
            other_data.append([' '.join(splits), score])
    other_data.append(['Model name', model_name])
    other_data.append(['Eval dataset name', eval_dataset_name])
    other_data.append(['Eval duration', duration])

    # Fill data and general_data
    for lab in labels:
        d = [lab]
        for met in unique_metrics:
            if (lab, met) in lab_met2score.keys():
                d.append(round(lab_met2score[(lab, met)], 3))
        bylabel_data.append(d)
    for typ in unique_types:
        d = [typ]
        for met in unique_metrics:
            d.append(round(type_met2score[(typ, met)], 3))
        general_data.append(d)

    # Prepare dataframes
    df_bylabel_data = pd.DataFrame.from_records(bylabel_data, columns=['label'] + list(unique_metrics))
    df_bylabel_data.sort_values('label')

    df_general_data = pd.DataFrame.from_records(general_data, columns=['type'] + list(unique_metrics))
    df_general_data.sort_values('type')

    df_other_data = pd.DataFrame.from_records(other_data, columns=['key', 'value'])
    df_other_data.sort_values('key')

    # Build excel file and save it
    writer = pd.ExcelWriter(os.path.join(output_dir, output_filename), engine='openpyxl')
    df_bylabel_data.to_excel(writer, sheet_name='By label', index=False)
    df_general_data.to_excel(writer, sheet_name='General', index=False)
    df_other_data.to_excel(writer, sheet_name='Other', index=False)
    writer.save()