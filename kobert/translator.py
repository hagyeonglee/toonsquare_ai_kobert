import os
import sys
import urllib.request
import logging
import json
import detector

try:
    PAPAGO_API_ID = os.environ['PAPAGO_API_ID']
    PAPAGO_API_SECRET = os.environ['PAPAGO_API_SECRET']

except KeyError:
    PAPAGO_API_ID = 'check papago api id'
    PAPAGO_API_SECRET = 'check papago api secret'

def translator(sentence):
    encText = urllib.parse.quote(sentence)
    langCode = detector.detector(sentence)
    data = "source="+langCode+"&target=ko&text=" + encText
    url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", PAPAGO_API_ID)
    request.add_header("X-NCP-APIGW-API-KEY", PAPAGO_API_SECRET)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if (rescode == 200):
        response_body = response.read()
        json_data = response_body.decode('utf-8')
        json_dict = json.loads(json_data)
        trans_result = json_dict['message']['result']['translatedText']
        # logging.info(json_data)
        # logging.info(json_dict['message'])
        # logging.info(json_dict['message']['result'])
        # logging.info(json_dict['message']['result']['translatedText'])
        logging.info(trans_result)
        
    else:
        logging.info("Error Code:" + rescode)

    return trans_result