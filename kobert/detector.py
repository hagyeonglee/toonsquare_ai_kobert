import os
import sys
import urllib.request
import logging
import json

try:
    PAPAGO_API_ID = os.environ['PAPAGO_API_ID']
    PAPAGO_API_SECRET = os.environ['PAPAGO_API_SECRET']

except KeyError:
    PAPAGO_API_ID = 'check papago api id'
    PAPAGO_API_SECRET = 'check papago api secret'

def detector(sentence):
    encQuery = urllib.parse.quote(sentence)
    data = "query=" + encQuery
    url = "https://naveropenapi.apigw.ntruss.com/langs/v1/dect"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID",PAPAGO_API_ID)
    request.add_header("X-NCP-APIGW-API-KEY",PAPAGO_API_SECRET)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()

    if(rescode==200):
        response_body = response.read()
        json_data = response_body.decode('utf-8')
        json_dict = json.loads(json_data)
        langCode = json_dict['langCode']
        # print(json_data)
        # print(json_dict)
        logging.info(langCode)

    else:
        logging.info("Error Code:" + rescode)

    return langCode