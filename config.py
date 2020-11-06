# Directories for project and models special tokens
import os
from os import path

BASE_DIR = os.environ.get('CELI_PROJECT')
DATASETS_DIR = path.join(BASE_DIR, 'datasets')
MODELS_DIR = path.join(BASE_DIR, 'models')

SPECIAL_TOKENS = {'[CLS]', '[SEP]'}
