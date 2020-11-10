# Directories for project and models special tokens
import os
from os import path

# You should consider to set up and environment variable in order to make
# this project working in other computers.
# BASE_DIR = os.environ.get('BASE_PROJECT_DIR')
#
# For now, let's use a normal variable
BASE_DIR = './'

DATASETS_DIR = path.join(BASE_DIR, 'datasets')
MODELS_DIR = path.join(BASE_DIR, 'models')

# Special tokens used for inference
SPECIAL_TOKENS = {'[CLS]', '[SEP]'}
