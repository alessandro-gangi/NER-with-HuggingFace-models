import os
from datetime import timedelta, datetime
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


def save_evaluation_result(df_scores: pd.DataFrame, df_conf_matrix: pd.DataFrame, df_errors: pd.DataFrame,
                           model_name: str, eval_dataset_name: str,
                           split_indexes: tuple, duration: float, output_dir: str, output_filename: str,
                           aggregations: dict, deleted_entities: list):
    """
    Write metric results on file.
    :param df_scores: pd.DataFrame
        dataframe with rows=labels and columns=scores
    :param df_conf_matrix: pd.DataFrame
        dataframe representing the confusion matrix
    :param df_errors: pd.Dataframe
        dataframe representing the errors made by model
    :param model_name: str
        Name of the model evaluated
    :param eval_dataset_name: str
        name of the dataset used for evaluation
    :param split_indexes: tuple of len(2)
        tuple containing lists of document indexes used for
        training (tuple[0]) and for evaluation (tuple[1])
    :param duration: float
        training duration time
    :param output_dir: str
        path to folder where result will be saved
    :param output_filename: str
        name of the results file that will be saved
    :param aggregations: dict
        entities aggregated ('key' entity aggregated into 'value' entity)
    :param deleted_entities: list
        entities that were not considered in model training (and evaluation)
    :return: void
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    today_date_str = datetime.now().strftime("%Y%m%d")
    other_data = [['model name', model_name], ['dataset name', eval_dataset_name],
                  ['eval duration', duration], ['date', today_date_str]]
    df_other_data = pd.DataFrame.from_records(other_data, columns=['key', 'value'])
    df_other_data.sort_values('key')

    df_indexes_data = pd.DataFrame({'train': split_indexes[0],
                                    'test': split_indexes[1]})

    df_ent_prep = pd.DataFrame(columns=['from', 'to'])
    if aggregations:
        df_ent_prep = df_ent_prep.append(pd.DataFrame.from_records([[e_src, e_targ] for e_src, e_targ
                                                                    in aggregations.items()],
                                                                   columns=['from', 'to']))
    if deleted_entities:
        df_ent_prep = df_ent_prep.append(pd.DataFrame.from_records([[ent, 'DELETED'] for ent in deleted_entities],
                                                                   columns=['from', 'to']))

    # Build excel file and save it
    writer = pd.ExcelWriter(os.path.join(output_dir, output_filename), engine='openpyxl')
    df_scores.to_excel(writer, sheet_name='Scores', index=False)
    df_conf_matrix.to_excel(writer, sheet_name='Confusion matrix', index=False)
    df_errors.to_excel(writer, sheet_name='Errors', index=False)
    if not df_ent_prep.empty:
        df_ent_prep.sort_values('from')
        df_ent_prep.to_excel(writer, sheet_name='Entities pre-processing', index=False)

    df_indexes_data.to_excel(writer, sheet_name='Split indexes', index=False)
    df_other_data.to_excel(writer, sheet_name='Other', index=False)
    writer.save()
