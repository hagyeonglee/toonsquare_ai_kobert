from flask import Flask, jsonify, request, render_template
from flask_ngrok import run_with_ngrok
from model import BertForEmotionClassification
from datasets import get_pretrained_model, Datasets
from pytorch_transformers.modeling_bert import BertConfig
import numpy as np
import torch

pretrained_model_path = './best_model/best_model.bin'
config_path = './best_model/bert_config.json'

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
# app._static_folder = './static'
run_with_ngrok(app)


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
    app.run()
#host='0.0.0.0',port=80
