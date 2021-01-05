HuggingFace for Named Entities Recognition
==============================
Finetune and evaluate HuggingFace models on custom datasets. Then make inference.

What Is This?
-------------

This is a Python 3.7 project for testing HuggingFace models performance on NER task. It's made of 2 different parts:
1. FINETUNING AND EVALUATION: chose a model, a training dataset and an evaluation dataset and see how good the 
model you finetuned works.
2. INFERENCE: Chose a model and a test document and see your model in action.

How to use?
-------------
* Install required libraries.
* Create an environment variable named "BASE_PROJECT_DIR" pointing at the root of this project (check config.py)

HOW TO FINETUNE AND EVALUATE:

Run

python finetune.py *model_name* *dataset_name*
(es: python finetune.py distilbert-base-cased train.txt)

to train and evaluate the model *model_name* on dataset *dataset_name* (dataset will be splitted)

You can train a model from scratch using one the standard HuggingFace models (as 'bert-base-cased', 
'dbmdz/bert-base-italian-cased') or you can train one of your previously saved models
(as 'bert-base-cased_yyyymmdd').

You can also just evaluate a model (an HugginFace standard one or any of yours) by running

python finetune.py *model_name* *dataset_name* -notrain

The following list shows all the command line parameters for *finetune* script:
* 'model *name*' - Name of a specific model previously saved inside "models" folder or name 
of an HuggingFace model'
* 'dataset *name*' - Name of a specific dataset present inside "datasets" folder. Both doccano (json1) 
and conll are supported
* '-tok *name*' - Name of a specific tokenizer (check HuggingFace list). If not provided, an automatic tokenizer will be used
* '-config *name*' - Name of a specific model configuration (check HuggingFace list). If not provided, an automatic configuration will be used
* '-notrain' - If set, training will be skipped
* '-noeval' - If set, evaluation phase will be skipped
* '-noplot' - If set, no charts will be plotted

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
