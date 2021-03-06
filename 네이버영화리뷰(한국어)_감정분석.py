# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="tJ6mtbuUVrST"
# # 네이버 리뷰 데이터를 활용한 한국어 감정 분석
# 네이버 영화 리뷰데이터(Naver Sentiment Movie Corpus,NSMC)를 활용해서 감정분석을 수행했습니다.  
# .   
# 1)
# 전처리 코드는 다음 github에서 가져와서 사용했습니다.
# 출처 : https://github.com/reniew/NSMC_Sentimental-Analysis/blob/master/notebook/NSMC_Preprocessing.ipynb  
# .  
# 2)
# 모델링은 IMDB에서 수행한 LSTM 으로 진행했습니다.
# -

pip list
pip install tensorflow --user
pip install konlpy

import sys
sys.version

import tweepy
import jpype

# + id="ahBLwH7pVsE-"
import os

import numpy as np
import pandas as pd

from datetime import datetime
import json
import re ## 정규표현식

from konlpy.tag import Okt # komoran, han, kkma

import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing import sequence

from tqdm.notebook import tqdm
# -

https://konlpy.org/ko/v0.5.2/morph/


# + [markdown] id="ts48shKOeF_B"
# ## 데이터 불러오기

# + id="KHBx7PuwWNTi"
train = pd.read_csv('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt', header=0, delimiter='\t' ,quoting=3)
test = pd.read_csv('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt', header=0, delimiter='\t' ,quoting=3)

# + colab={"base_uri": "https://localhost:8080/", "height": 383} executionInfo={"elapsed": 11767, "status": "ok", "timestamp": 1611885530842, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="jYsgfARTY6Lb" outputId="a5f44fdc-5863-413e-f170-72b9cc9280cf"
display(train.head())  ## train data
display(test.head())   ## test data

## label 긍부정 정보가 들어가있음

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11129, "status": "ok", "timestamp": 1611885530843, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="br0HdVrdY4vr" outputId="6fb265a9-a32a-4692-e96c-3a80695ffcfa"
train.shape, test.shape


# + [markdown] id="nMdHcJrWeJWq"
# ## 데이터 전처리

# + id="0oT3lGwpWxJf"
def preprocessing(review, okt, remove_stopwords = False, stop_words = [], test = False):
    # 함수의 인자는 다음과 같다.
    # review : 전처리할 텍스트
    # okt : okt 객체를 반복적으로 생성하지 않고 미리 생성후 인자로 받는다.
    # remove_stopword : 불용어를 제거할지 선택 기본값은 False
    # stop_word : 불용어 사전은 사용자가 직접 입력해야함 기본값은 비어있는 리스트
    
    # 1. 한글 및 공백을 제외한 문자 모두 제거.
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", review) 
    ## re.sup [^가-힣ㄱ-ㅎㅏ-ㅣ\\s]이런 것을 찾아서 null값으로 대체하겠다 (replace) 
    ## 가-힣 = A-Z 와 같은 의미이다. ㄱ-ㅎ, ㅏ-ㅣ
    
    # 2. okt 객체를 활용해서 형태소 단위로 나눈다.
    word_review = okt.morphs(review_text, stem=True)
    ## 한글과 공백만 남긴 것에대해 형태소 분류를 한다
    
    if test:
        print(review_text)
        print(word_review)

    if remove_stopwords: ## 불용어 제거
        
        # 불용어 제거(선택적)
        word_review = [token for token in word_review if not token in stop_words]

    return word_review


# + colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"elapsed": 425, "status": "ok", "timestamp": 1611885754992, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="LqW3KS3z3bL0" outputId="edf8d98d-ef87-423c-9a25-e7ce9ab90dc6"
sample_review = train['document'][0]
sample_review
# -

import platform 
print(platform.architecture())

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 623, "status": "ok", "timestamp": 1611885758701, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="L9q4V-RO3S_c" outputId="71800dbb-93c9-4dc4-d052-a8184b579c4d"
## 불용어에 대해 stop_words로 저장함.
stop_words = [ '은', '는', '이', '가', '하', '아', '것', '들',
              '의', '있', '되', '수', '보', '주', '등', '한']
preprocessing(sample_review, Okt(), remove_stopwords = True, stop_words=stop_words, test = True)


## sample_review에서 한글이랑 공백만 남기고 제거, 불용어 제거, 형태소 제거

# +
## 행태소 분류
## 아 더빙 진짜 짜증나네요 목소리 <<< 가 형태소 분류로 아래와 처럼 출력됨.
## ['아', '더빙', '진짜', '짜증나다', '목소리']

# +
## 불용어 제거

## ['아', '더빙', '진짜', '짜증나다', '목소리'] 에서 불용어 제거하면 아래처럼 출력됨
## ['더빙', '진짜', '짜증나다', '목소리']

# + colab={"base_uri": "https://localhost:8080/", "height": 115, "referenced_widgets": ["3a5243871ce54f21a4169c128df27da0", "7ef50d251fe24059a1b3fc2d5b1a8c36", "af717e8773b2460a943ac2b19a8ff153", "eebb424d5d7e4ac5a7a29ea36f28932e", "08636d19300f4ca69dd917577db6cbf3", "0c0b30994ccf42a8be2dbcc4e4c2cf03", "f275bd9b4aa64457b063b98ff5f3982e", "ab688ed7e5ba404aa8843e608582bdc0", "4ce61847174c4c62960902a855fdf7a1", "d6b87061815e4efea57f6b2731310349", "08ed07ad527046469e08ff2cf3d529e2", "4389e1fa83c94abf8c3ba894265a7171", "2800d0d7db40410d91ebe8dcd3b4fb81", "a366e000317844eaa61b1ce1de36a166", "975ab15d5d514fd0b1a8c51b0d7895d0", "e19df99fe25d401582cc363df39035e8"]} executionInfo={"elapsed": 581966, "status": "ok", "timestamp": 1611886425124, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="j2ZfzrInXEWJ" outputId="f808ff56-5152-4b5a-8bb9-696514363d55"
stop_words = [ '은', '는', '이', '가', '하', '아', '것', '들','의', '있', '되', '수', '보', '주', '등', '한']
okt = Okt()  ## 한국어 정보처리를 위한 파이썬 패키지
clean_review = [] ## 정리된 ['더빙', '진짜', '짜증나다', '목소리'] 이러한 list 들이 들어옴.
clean_review_test = []

for review in tqdm(train['document']):
    ## train data에 for문을 돌려서 review 값을 받음
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str: ## review type str 이면 append 해죽
        clean_review.append(preprocessing(review, okt, remove_stopwords = True, stop_words=stop_words)) ## okt를 넣어줘서 전처리함
    else:  
        clean_review.append([])  ## str아닌 경우 결측치로 넣겠다.

for review in tqdm(test['document']):
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        clean_review_test.append(preprocessing(review, okt, remove_stopwords = True, stop_words=stop_words))
    else:
        clean_review_test.append([])

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 138813, "status": "ok", "timestamp": 1611886425125, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="0W9fPy4zXTGw" outputId="1e3e9e5e-84a2-42fc-daf1-7f09c19d90dd"
print(len(clean_review))
print(len(clean_review_test))

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 141793, "status": "ok", "timestamp": 1611886428331, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="esgIQAHdXVfQ" outputId="2c357c8b-56c6-4f0b-ec55-1b3c52647f64"
## Tokenizer()를 clean_review 기준으로 fit 해주는 것
## ['더빙', '진짜', '짜증나다', '목소리'] list들을 [1, 2, 3, 4]형태로 바꿔준 것이 text_sequences 이다
tokenizer = Tokenizer()   
tokenizer.fit_on_texts(clean_review) # 단어 인덱스 구축
text_sequences = tokenizer.texts_to_sequences(clean_review) # 문자열 -> 인덱스 리스트
                                                            # '나는 천재다 나는 멋있다' -> [1, 2, 1, 3]
                                                            # sequnce 4개를 숫자로 바꿔줘야함. 나는 = 1, 천재다 = 2, 멋있다 = 3

## word와 idex를 딕셔너리 형태로 만드는 것
word_vocab = tokenizer.word_index # 딕셔너리 형태
print("전체 단어 개수: ", len(word_vocab)) # 전체 단어 개수 확인

## 43756 학습 하는 데이터의 단어 개수

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 142305, "status": "ok", "timestamp": 1611886429027, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="iBN97IQFXX_A" outputId="1c9c46c9-cffc-42c4-e451-b060e6a917ab"
MAX_SEQUENCE_LENGTH = 50 # 문장 최대 길이

X_train = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post') 
# 문장의 길이가 50 단어가 넘어가면 자르고, 모자르면 0으로 채워 넣는다.
y_train = np.array(train['label']) # 각 리뷰의 감정을 넘파이 배열로 만든다.

print('Shape of input data tensor:', X_train.shape) # 리뷰 데이터의 형태 확인 # 150000개 리뷰, 하나의 문장(sequence가 50개)
print('Shape of label tensor:', y_train.shape) # 감정 데이터 형태 확인

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 143598, "status": "ok", "timestamp": 1611886430512, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="3uwv-C52XamA" outputId="b9055824-73d7-49b4-aa27-45affcdd98ec"
tokenizer_test = Tokenizer()
tokenizer_test.fit_on_texts(clean_review_test)
text_sequences_test = tokenizer_test.texts_to_sequences(clean_review_test)

word_vocab_test = tokenizer_test.word_index # 딕셔너리 형태
print("전체 단어 개수: ", len(word_vocab_test)) # 전체 단어 개수 확인

## test set에는 단어가 26778개가 있다.

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 143516, "status": "ok", "timestamp": 1611886430797, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="fl_17663Xcsn" outputId="1ba98978-3c00-4416-8b2d-c97fe4b2a4dc"
MAX_SEQUENCE_LENGTH = 50 # 문장 최대 길이

X_test = pad_sequences(text_sequences_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post') # 문장의 길이가 50 단어가 넘어가면 자르고, 모자르면 0으로 채워 넣는다.
y_test = np.array(test['label']) # 각 리뷰의 감정을 넘파이 배열로 만든다.

print('Shape of input data tensor:', X_test.shape) # 리뷰 데이터의 형태 확인 ## # 50000개 리뷰, 하나의 문장(sequence가 50개)
print('Shape of label tensor:', y_test.shape) # 감정 데이터 형태 확인

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 143099, "status": "ok", "timestamp": 1611886430797, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="6Vyp9zKChvDu" outputId="7744941a-7e84-44ac-cbd9-71bc8b5131e1"
X_train
## [  463,    20,   265, ...,     0,     0,     0] << [더빙, 진짜, 짜증나 ....]가 숫자로 변한것.

# + [markdown] id="NgeOjOHgeNLC"
# ## 모델 구축

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 81760, "status": "ok", "timestamp": 1611886433010, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="Z2Gv7oGNdqvD" outputId="73f90d74-cf3d-4665-ac97-4577dccbf6aa"
## TensorFlow에 keras가 이젠 포함됨

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

## 딥러닝은 구조적인 모델이기에 층 별로 쌓아야함.
## 딥러닝(다층) >  우리가 직접 층을 쌓음

model = Sequential()
## 가장 기본적인 딥러닝 구조를 만들 때 사용하는 방법
## 모델에 층을 하나 씩 붙이기 때문에 처음에 객체를 Sequential로 하고

model.add(Embedding(len(word_vocab)+1, 400)) # (단어집합의 크기, 임베딩 후 벡터 크기)
## Embedding : 463개를 400개의 vector를 만들어줌
## 입력 층
## word_vocab : 각 각을 vocab으로 만들고 길이를 재서 vector로 만들고 그 vector의 길이를 400개로 Embedding(압축) 해준다
model.add(LSTM(128))
## 출력 층
## 128개로 히든레이어를 잡는다
model.add(Dense(1, activation = 'sigmoid')) # 0 or 1로 이진분류이므로 시그모이드 함수를 사용
## sigmoid = loistic
## 마지막에 1 ~ 0 인지 긍정 / 부정으로 확률값 계산
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
## compile은 모델의 세부 설정이다.
## 모델에 층을 다 쌓고 나면 모델의 세부설정을 하는데 세부설정은 3가지 
## (최적화방법, 오차계산방식 : cost계산하는 방법은 logloss를 사용, 평가방식 : 정확도)

# 이진 분류이므로 손실함수는 binary_crossentropy 사용, 에폭마다 정확도를 보기 위해 accuracy 적용
print(model.summary()) #모델 아키텍처 출력

## model.summary 모델의 구조를 요약하여출력

# + [markdown] id="lXmz9cNjeP3a"
# ## 모델 학습

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 645246, "status": "ok", "timestamp": 1611886997871, "user": {"displayName": "wonjae lee", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjMlwCQjMGFZtS8hbmZRdBqju40soJi1q--Prlt=s64", "userId": "02946067935890729436"}, "user_tz": -540} id="kgH78W1bdz5x" outputId="5b946ab4-cb83-489a-8c27-2b57967bad4c"
model.fit(X_train, y_train, epochs =1, batch_size = 60, validation_split=0.2)
## epochs : data를 몇번 학습시킬 것 인지를 나타냄.
    ## 같은 데이터 일수록 학습을 여러번 하게되면 성능이 좋아지냐는 
    ## 첫번째 학습은 w 초기값으로 학습을 시작하고, 학습이 끝나면 w는 학습된 값이다.
    ## 두번재 학습은 1차 학습한 w 값에서 학습을 시작하고 업데이트를 하는 것이기 때문에 성능은 더 좋아진다.
    ## 하지만 너무 많이 하면 과적합이 일어나기에 학습을 반복시 val_score과, weight를 기록하여 좋은 값을 찾아 선택한다.

## batch_size : 데이터 셋을 60개 를 갖고 예측값 60개를 출력해서 cost 60개를 계산해서 w 한번을 없데이트 하자.
                ## 150000만개 데이터에서 60개를 갖고 w 업데이트를 하니 w를 2500번 업데이트를 할 수 있다.
        ## batch : 한번에 데이터를 w 업데이트를 하기위한 데이터의 분량
        ## 150000개를 한번에 cost 값을 계산해서 한개의 w값을 찾는 것 보다 60개씩 나눠서 2500번 cost를 구하고 w를 업데이트를 하는 것이 더 좋다.


## X_train, y_train에 리뷰가 들어가서 긍정인지 부정사이의 w를 학습해서 새로운 입력값이 들어오면 긍정인지 부정인지 확률값으로 예측함.

## 딥러닝에서 LSTM 은 확률값으로 예측하고 비선형이다.


##  0.2니까 train data는 120000개고 데이터를 60개씩 돌리니까 2000번 w이고 그 반복 epochs을 1번 하기에 2000이다.
## 120000/60 = 2000 > 2000 * 1(epochs) = 2000

# + [markdown] id="WGFTVh8-eRua"
# ## 모델 검증

# + colab={"base_uri": "https://localhost:8080/"} id="z5kOTiLyd2fy" outputId="cebde8af-4c3d-4395-8df3-91a075b461fc"
from sklearn.metrics import accuracy_score



y_train_predclass = model.predict(X_train)
classes_x=np.argmax(y_train_predclass,axis=1)

y_test_predclass = model.predict(X_test)
classes_y=np.argmax(y_test_predclass,axis=1)

# predict_x=model.predict(X_test) 
# classes_x=np.argmax(predict_x,axis=1)

print("Train Accuracy: {}".format(round(accuracy_score(y_train, classes_x),3)))
print("Test Accuracy: {}".format(round(accuracy_score(y_test, classes_y),3)))



## model.predict_classes는 업데이트되어 사라짐.

# y_train_predclass = model.predict_classes(X_train)
# y_test_predclass = model.predict_classes(X_test)

# print("Train Accuracy: {}".format(round(accuracy_score(y_train, y_train_predclass),3)))
# print("Test Accuracy: {}".format(round(accuracy_score(y_test, y_test_predclass),3)))
# -

y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# + id="kBrG8zRhyDS1"
'ㄴㅏㄴㅡㄴㅂㅐㄱㅗㅍㅡㄷㅏ' -> '나는배고프다' # hangul_utils . jamo_join 를 통해 나는배고프다로 변경할 수 있다.
