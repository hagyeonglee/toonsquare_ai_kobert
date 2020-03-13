from flask import Flask, jsonify, request, render_template
from model import BertForEmotionClassification
from datasets import get_pretrained_model, Datasets
from pytorch_transformers.modeling_bert import BertConfig

import numpy as np
import torch

# from flask_ngrok import run_with_ngrok

import sys
import shutil
import os
import logging
import utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import constant

from azure.storage.file import FileService

REAL_PATH = os.path.realpath(__file__)
DIRNAME = os.path.dirname(REAL_PATH)

MODEL_ABS_PATH        = os.path.join(DIRNAME, constant.MODEL_DIR_PATH)
MODEL_ZIP_ABS_PATH        = os.path.join(DIRNAME, constant.MODEL_ZIP_NAME)

CONFIG_FILE_ABS_PATH = os.path.join(DIRNAME, constant.BERT_CONFIG_NAME)


try:
    AZURE_STORAGE_ACCOUNT_NAME = os.environ['AZURE_STORAGE_ACCOUNT_NAME']
    AZURE_STORAGE_ACCOUNT_KEY = os.environ['AZURE_STORAGE_ACCOUNT_KEY']
    AZURE_STORAGE_NAME = os.environ['AZURE_STORAGE_NAME']
except KeyError:
    AZURE_STORAGE_ACCOUNT_NAME ="check Account name"
    AZURE_STORAGE_ACCOUNT_KEY = "check Account key"
    AZURE_STORAGE_NAME = "check storage name"

#model, config files download and unzip
try:
    FILE_SERVICE = FileService(account_name=AZURE_STORAGE_ACCOUNT_NAME, account_key=AZURE_STORAGE_ACCOUNT_KEY)
    logger.debug("MODEL_ABS_PATH : %s", MODEL_ABS_PATH)
    if os.path.exists(MODEL_ABS_PATH):
        shutil.rmtree(MODEL_ABS_PATH)

    FILE_SERVICE.get_file_to_path(AZURE_STORAGE_NAME, constant.STORAGE_BERT_SENTIMENT_DIR, constant.MODEL_ZIP_NAME, MODEL_ZIP_ABS_PATH)
    FILE_SERVICE.get_file_to_path(AZURE_STORAGE_NAME, constant.STORAGE_BERT_SENTIMENT_DIR, constant.BERT_CONFIG_NAME, CONFIG_FILE_ABS_PATH)
    shutil.unpack_archive(MODEL_ZIP_ABS_PATH, extract_dir=DIRNAME)

except Exception as e:
    logger.critical("Unexpected error : %s", e)
    sys.exit()

# print(DIRNAME)

pretrained_model_path = os.path.join(MODEL_ABS_PATH,constant.MODEL_BIN_NAME)
# print(pretrained_model_path)

config_path = os.path.join(DIRNAME,constant.BERT_CONFIG_NAME)
# print(config_path)

pretrained = torch.load(pretrained_model_path,map_location='cpu')
bert_config = BertConfig(config_path)
bert_config.num_labels = 7

model = BertForEmotionClassification(bert_config)
model.load_state_dict(pretrained, strict=False)
model.eval()
softmax = torch.nn.Softmax(dim=1)

tokenizer, vocab = get_pretrained_model('etri')

# '공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오'
# 'angry', 'surprise', 'angry', 'sadness', 'neutral', 'joy', 'disgust'
obj = dict()
emotion = ['scare', 'surprise', 'angry', 'sadness', 'neutral', 'joy', 'disgust']

def get_prediction(sentence):
    sentence = Datasets.normalize_string(sentence)
    sentence = tokenizer.tokenize(sentence)
    sentence = tokenizer.convert_tokens_to_ids(sentence)
    sentence = [vocab['[CLS]']] + sentence + [vocab['[SEP]']]

    output = model(torch.tensor(sentence).unsqueeze(0))
    output_softmax = softmax(output)[0]
    max_out = emotion[output_softmax.argmax()]
    argidx = output_softmax.argsort(descending=True)
    result = {emotion[i]: round(output_softmax[i].item(), 3) for i in range(len(emotion))}
    sorted_result = {emotion[i]: round(output_softmax[i].item(), 3) for i in argidx}
    return max_out, result, sorted_result


app = Flask(__name__)
# run_with_ngrok(app)

@app.route('/')
def test():
    return render_template('post.html')

@app.route('/post', methods=['POST'])
def post():
    sentence = request.form['input']
    max_out, result, sorted_result = get_prediction(sentence)
    obj['prediction'] = {
        'emotion': max_out,
        'data': result
    }
    return obj

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
