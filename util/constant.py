import colorsys
import datetime
import os
from enum import Enum

now_instance = datetime.datetime.now()

TIME_TAG = now_instance.strftime('%Y%m%d%H%M%S.')

VERSION_FILE = 'version.json'
STORAGE_BERT_SENTIMENT_DIR = "bert_sentiment"

# model deploy
MODEL_ZIP_NAME = 'model.zip'
MODEL_DIR_PATH = 'model'
MODEL_BIN_NAME = 'best_model.bin'

#bert_config.json 
BERT_CONFIG_NAME = 'bert_config.json'