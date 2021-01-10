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

ENTITIES_AGGREGATIONS = {
    'I.I. Ai Common Data: jobsector': 'I.I. Av Common Data: job title',
    'Pers Det: Residence Place': 'I.I. Aii Com Data: city/town',
    'D.I. Aiii Pers Det: Birthplace': 'I.I. Aii Com Data: city/town',
    'Region': 'I.I. E Country',
    'Regional Provenance': 'I.I. G Nationality',
    'D.I. Aiv Pers Det: DOB': 'I.I. Avii Common Data: Date',
}

# We can also define a list of entities to delete (not consider) before we train the model
ENTITIES_TO_FILTER = ['DOUBT', 'Common data: Economic']
