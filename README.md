HuggingFace for Named Entities Recognition
==============================
Finetune and evaluate HuggingFace models on custom datasets. Then make inference.

What Is This?
-------------

This is a Python 3.6 project for testing HuggingFace models performance on NER task. It's made of 2 different parts:
1. FINETUNING AND EVALUATION: chose a model, a training dataset and an evaluation dataset and see how good the 
model you finetuned works.
2. INFERENCE: Chose a model and a test document and see your model in action.

How to use?
-------------
* Install required libraries.
* Create an environment variable named "CELI_PROJECT" pointing at the root of this project (check config.py)

HOW TO FINETUNE AND EVALUATE:

Run

python finetune.py *model_name* -trainset *trainingset_name* -evalset *evaluationset_name*
(es: python finetune.py distilbert-base-cased -trainset train.txt -evalset eval.txt)

to train the model *model_name* on dataset *trainingset_name* and then evaluate the model 
on *evaluationset_name* dataset.

You can train a model from scratch using one the standard HuggingFace models (as 'bert-base-cased', 
'dbmdz/bert-base-italian-cased') or you can train one of your previously saved models
(as 'bert-base-cased_yyyymmdd').

You can also just evaluate a model (an HugginFace standard one or one of yours) by running

python finetune.py *model_name* -notrain -evalset *evaluationset_name*

The following list shows all the command line parameters for *finetune* script:
* 'model *name*' - Name of a specific model previously saved inside "models" folder or name 
of an HuggingFace model'
* '-trainset *name*' - Name of a specific training dataset present inside "datasets" folder. Training dataset 
  should contain one word and one label per line; there should be an empty line between two sentences
* '-evalset *name*' - Name of a specific evaluation dataset present inside "datasets" folder. Evaluation
dataset should contain one word and one label per line; there should be an empty line between two sentences
* '-tok *name*' - Name of a specific tokenizer (check HuggingFace list). If not provided, an automatic tokenizer will be used
* '-config *name*' - Name of a specific model configuration (check HuggingFace list). 
If not provided, an automatic configuration will be used
* '-notrain' - If set, training will be skipped
* '-noeval' - If set, evaluation phase will be skipped
* '-noplot' - If set, no charts will be plotted
* '-traineval' - If set, model wont be evaluated during training

Other specific arguments:
* '-maxseqlen *value*' - int - Value used by tokenizer to apply padding or truncation to sequences. If not provided an automatic 
value will be chosen according to the model used
* '-epochs *value*' - int - Number of epochs during training. If not provided, 2 epochs will be used.
* '-warmsteps *value*' - int -Number of warm-up steps before training. If not provided, 500 steps will be used.
* '-wdecay *value*'- float - Weight decay to use during training. If not provided, 0.01 decay will be used.
* '-trainbatch *value*' - int - Per device batch size during training. If not provided, 32-batch will be used.
* '-evalbatch *value*' - int - Per device batch size during evaluation. If not provided, 64-batch will be used.
* '-logsteps *value*' - int - Number of training steps between 2 logs. If not provided, 100 steps will be used.


HOW TO MAKE INFERENCE:

Run

python inference.py *model_name* *document_name*
(es: python inference.py distilbert-base-cased_yyyymmdd test.txt)

to use the model *model_name* to infere named entities of *document_name* document and save a new,
annotated copy of it.

The following list shows all the command line parameters for *inference* script:
* 'model *name*' - Name of a specific model previously saved inside "models" folder or name 
of an HuggingFace model'
* 'doc *name*' - Name of a specific .txt document present inside "datasets" folder. The document
should contain only plain text (without labels). Sequences should be separated by "\n"
* '-noscores' - If set, confidence scores won't be saved in the annotated output document


Other infos
-------------
* Only HuggingFace models with Fast Tokenizers are supported beacuse of the way the data is
encoded and aligned with labels.
