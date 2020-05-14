# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from selenium import webdriver
import os
import time
import pandas as pd

from model import BertForEmotionClassification
from datasets import get_pretrained_model, Datasets
from pytorch_transformers.modeling_bert import BertConfig
import numpy as np
import torch

import sys
import shutil
import logging

import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.chdir('/home/ellie/Downloads/chromedriver_linux64') #chrome driver가 저장된 디렉토리
# chromedriver = '/home/ellie/Downloads/'
# driver = webdriver.Chrome(chromedriver)

base_url = 'https://comic.naver.com/webtoon/weekday.nhn'
# driver.get(base_url)
#  driver 실행
def drive(url):
    driver = webdriver.Chrome('./chromedriver') #driver 객체 불러옴
    driver.implicitly_wait(3) # 3초 후에 작동하도록
    driver.get(url) #url에 접속
    html = driver.page_source #현재 driver에 나타난 창의 page_source(html) 가져오기
    soup = BeautifulSoup(html, 'html.parser') #html 파싱(parsing)을 위해 BeautifulSoup에 넘겨주기
    return driver, soup

#웹툰 기본 페이지에서 데이터 가져오기
driver, soup = drive(base_url)
driver.close()

#가져온 데이터 파싱, id, 요일, 이름
title = soup.select('.title')
t_IDs=list(map(lambda x: x.get('href').split('titleId=')[1].split('&')[0], title))
t_weekdays = list(map(lambda x: x.get('href').split('weekday=')[1], title))
t_names = list(map(lambda x: x.text ,title))

#크롤링이 잘 되었나 확인하기 위함, 총 웹툰 수
print('t_IDs: ',len(t_IDs))
print('t_weekdays: ',len(t_weekdays))
print('t_names: ',len(t_names))
# arr = [t_IDs[101],t_weekdays[101],t_names[101]]
# for i in arr:
#     print(i)
    # # print(type(i))
    # print(unicode(i.encode('cp949')))
    # # print(i.decode('EUC-KR'))

input_name = input('크롤링할 웹툰의 이름을 입력 : ')
# '유미의 세포들'

#웹툰 이름으로 id와 weekday 반환
def find_id_weekday(name,t_names,t_IDs,t_weekdays,start_idx = 0):
    try:
        idx = t_names.index(name)
    except:
        print('찾는 웹툰이 없습니다.')
        return
    return t_IDs[idx], t_weekdays[idx]

ID, weekday = find_id_weekday(input_name,t_names,t_IDs,t_weekdays,start_idx = 0)

# print(type(input_name)) #<class 'str'>
# print(type(ID))
# print(type(weekday))
print(input_name+"의 ID : "+ID)
# print(input_name+"의 weekday : "+weekday)

# episode 개수 세기
def episode_count(ID, weekday):
    url = base_url.split('weekday')[0] + 'list.nhn?titleId={0}&weekday={1}'.format(ID, weekday)
    # print(url)
    driver, soup = drive(url)
    driver.close()
    res = soup.select('.title')[0].select('a')[0].get('href').split('no=')[1].split('&')[0]

    return res

# res = episode_count(651673, 'wed')  #유미의 세포들 총 에피소드 수
res = episode_count(ID, weekday)
print(input_name+" 의 총 에피소드 수 : " + res)

#0부터 끝까지 회차의 모든 해당 웹툰의 best 댓글을 crawling하는 함수
def all_comment_crawler(name, start_idx=0):
    id_num, weekday = find_id_weekday(name, t_names, t_IDs, t_weekdays, start_idx=start_idx)
    cnt = int(episode_count(id_num, weekday))

    comments = []
    proceed = -1  # 진행 상태 표시 위함, 처음에 0보다 작아야 0%가 표시 됨

    driver, _ = drive(base_url)  # driver만 먼저 열어 놓음. for문 돌면서 url만 바꿔줄 것임
    print(name+' 크롤링 진행중...')
    # for i in range(1, cnt + 1):
    for i in range(1, cnt + 1):
        percentage = int((i / cnt) * 100)
        if percentage % 10 == 0 and percentage > proceed:  # 진행상황 표시
            proceed = percentage
            print(proceed, '% 완료')
        url = 'https://comic.naver.com/comment/comment.nhn?titleId={0}&no={1}#'.format(id_num, str(i))
        # driver.implicitly_wait(3)
        time.sleep(1.5)
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        comments += list(map(lambda x: x.text, soup.select('.u_cbox_contents')))

    driver.close()
    print('crawling finished')
    return comments
#

#원하는 회차의 웹툰의 best 댓글을 crawling하는 함수
def episode_comment_crawler(name, episode_num):
    id_num, weekday = find_id_weekday(name, t_names, t_IDs, t_weekdays, start_idx=episode_num)
    cnt = 1

    comments = []
    proceed = 0  # 진행 상태 표시 위함, 처음에 0보다 작아야 0%가 표시 됨

    driver, _ = drive(base_url)  # driver만 먼저 열어 놓음. for문 돌면서 url만 바꿔줄 것임
    print(name+" ep."+str(episode_num)+' 크롤링 진행중...')
    # for i in range(1, cnt + 1):
    # for i in range(1, 5):
    percentage = int((1 / cnt) * 100)
    # percentage % 10 == 0 and
    # if percentage > proceed:  # 진행상황 표시
    proceed += percentage
    print(proceed, '% 완료')
    url = 'https://comic.naver.com/comment/comment.nhn?titleId={0}&no={1}#'.format(id_num, str(episode_num))
    # driver.implicitly_wait(3)
    time.sleep(1.5)
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    comments += list(map(lambda x: x.text, soup.select('.u_cbox_contents')))

    driver.close()
    print('crawling finished')
    return comments
#

#원하는 회차의 범위의 웹툰의 best 댓글을 crawling하는 함수
def select_episode_comment_crawler(name, episode_start,episode_end):
    id_num, weekday = find_id_weekday(name, t_names, t_IDs, t_weekdays, start_idx=episode_start)
    cnt = int(episode_end)-int(episode_start)+1

    comments = []
    proceed = 0  # 진행 상태 표시 위함, 처음에 0보다 작아야 0%가 표시 됨

    driver, _ = drive(base_url)  # driver만 먼저 열어 놓음. for문 돌면서 url만 바꿔줄 것임
    print(name+" ep."+str(episode_start)+' ~ '+str(episode_end)+' 크롤링 진행중...')
    # for i in range(1, cnt + 1):
    for i in range(episode_start, episode_end+1):
        percentage = int((1 / cnt) * 100)
        # percentage % 10 == 0 and
        # if percentage > proceed:  # 진행상황 표시
        proceed += percentage
        print(proceed, '% 완료')
        url = 'https://comic.naver.com/comment/comment.nhn?titleId={0}&no={1}#'.format(id_num, str(i))
        # driver.implicitly_wait(3)
        time.sleep(1.5)
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        comments += list(map(lambda x: x.text, soup.select('.u_cbox_contents')))

    driver.close()
    print('crawling finished')
    return comments
#

#원하는 개의 웹툰의 best 댓글을 crawling하는 함수
def number_comment_crawler(name, comments_number):
    id_num, weekday = find_id_weekday(name, t_names, t_IDs, t_weekdays, start_idx=random.randint(1, int(res)))
    cnt = comments_number

    need_pagenum = int(comments_number/15)+1
    rest_pagenum = int(comments_number % 15 )

    print(need_pagenum)
    print(rest_pagenum)

    comments = []
    proceed = -1  # 진행 상태 표시 위함, 처음에 0보다 작아야 0%가 표시 됨

    driver, _ = drive(base_url)  # driver만 먼저 열어 놓음. for문 돌면서 url만 바꿔줄 것임
    print(name+' 크롤링 진행중...')
    # for i in range(1, cnt + 1):
    for i in range(1, need_pagenum + 1):
        percentage = int((i / cnt) * 100)
        if percentage % 10 == 0 and percentage > proceed:  # 진행상황 표시
            proceed = percentage
            # print(proceed, '% 완료')
        url = 'https://comic.naver.com/comment/comment.nhn?titleId={0}&no={1}#'.format(id_num, str(random.randint(1, int(res))))
        # driver.implicitly_wait(3)
        time.sleep(1.5)
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        comments += list(map(lambda x: x.text, soup.select('.u_cbox_contents')))
        # while (len(comments)==comments_number):
    j = 1
    for j in range(15-rest_pagenum):
        comments.pop()
        j += 1
            # print(j)


    driver.close()
    print('crawling finished')
    return comments
print("---------------------------------------")
print(" 1 : input episode (all | number of episode)\n 2 : episode range \n 3 : number of comments")
print("---------------------------------------")

mode = int(input("원하는 크롤링 모드를 선택하세요 : "))

if(mode==1):
    episode_num = input("원하는 에피소드 회차를 입력하세요 : ")
    # print(type(episode_num)) #str -> int()처리 필요
    if (episode_num == 'all'):
        comments = all_comment_crawler(input_name)
    else:
        episode_num = int(episode_num)
        comments = episode_comment_crawler(input_name, episode_num)
elif(mode==2):
    episode_start = int(input("원하는 에피소드 회차를 입력하세요(시작) : "))
    # print(type(episode_num)) #str -> int()처리 필요
    episode_end = int(input("원하는 에피소드 회차를 입력하세요(끝) : "))
    # print(type(episode_num)) #str -> int()처리 필요
    comments = select_episode_comment_crawler(input_name,episode_start,episode_end)
elif(mode==3):
    comments_number = int(input("원하는 댓글 개수를 입력하세요 : "))
    comments = number_comment_crawler(input_name, comments_number)


#수집된 댓글 수
print("수집된 댓글 수 : " + str(len(comments)))
# print(len(comments))

os.chdir('/home/ellie/toonsquare_ai/toonsquare_ai_kobert/kobert')
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
print(THIS_FOLDER)

#emotion predict
pretrained_model_path = './best_model/best_model.bin'
config_path = './best_model/bert_config.json'

pretrained = torch.load(pretrained_model_path, map_location='cpu')
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
emotion = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']

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
#
#추출한 댓글 저장 위해 현재 working directory 변경, 저장할 폴더 위치로 지정하면 된다.
# os.chdir('/home/ellie/webtoon_comments')
os.chdir('./data/webtoon_comments')

#추출한 댓글 저장 파일 이름 설정
# if (episode_num == ""):
#     file = open(input_name + "_ep." + str(episode_start) +" ~ "+ str(episode_end) + '_comments.csv', 'w', encoding='utf-8')
# else:
#     file = open(input_name + "_ep." + str(episode_num) + '_comments.csv', 'w', encoding='utf-8')

if(mode==1):
    file = open(input_name + "_ep." + str(episode_num) + '_comments.csv', 'w', encoding='utf-8')
elif(mode==2):
    file = open(input_name + "_ep." + str(episode_start) +" ~ "+ str(episode_end) + '_comments.csv', 'w', encoding='utf-8')
elif(mode==3):
    file = open(input_name + "_num." + str(comments_number) + '_comments.csv', 'w', encoding='utf-8')


#csv 파일의 column 이름 설정
index_num = 0
file.write("\""+"\""+","+"\""+"Sentence"+"\""+","+"\""+"Emotion"+"\""+'\n')
for cmt in comments:
    max_out, result, sorted_result = get_prediction(cmt)
    # "\""
    file.write("\""+str(index_num)+"\""+","+"\""+cmt+"\""+","+"\""+max_out+"\""+'\n')
    index_num += 1
    comments_df = pd.DataFrame(dict(Sentence=cmt, Emotion=max_out),columns =["", "Sentence","Emotion"],index= [index_num+1])
# print(comments_df.columns)
    # print(comments_df.shape)

file.close()
# sep="\t"
# comments_txt = pd.read_csv('/home/ellie/webtoon_comments/'+input_name+'_comments.csv',sep=',',engine='python',encoding='utf-8')
# print(comments_txt.shape)
# for emotion_pred in emotions:

# ,'emotion','pred'
# comments_df = pd.DataFrame([comments, max_out], columns=['Sentence','Emotion'])
# # comments_df.index = comments_df.index + 1
# comments_df.to_csv(f'naverwebtoon_{input_name}.csv',mode='w',encoding='utf-8-sig',header=True,index=True)

