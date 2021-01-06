# Directories for project and models special tokens
import os
from os import path

# You should consider to set up and environment variable in order to make
# this project working in other computers.
# BASE_DIR = os.environ.get('CELI_PROJECT')
#
# For now, let's use a normal variable
BASE_DIR = './'

DATASETS_DIR = path.join(BASE_DIR, 'datasets')
MODELS_DIR = path.join(BASE_DIR, 'models')

# Special tokens used for inference
SPECIAL_TOKENS = {'[CLS]', '[SEP]'}

#
#   Entities manipulation
#

# We can decide which entities to merge by editing the following dictionary
# Ex: if we want to merge the entities 'ent1' into 'ent2' we write
# {'ent1': 'ent2'}

ENTITIES_AGGREGATION = {
    'a': 'b',
    'c': 'd'
}

# We can also define a list of entities to delete before we train the model
ENTITIES_TO_DELETE = ['k', 'l']
