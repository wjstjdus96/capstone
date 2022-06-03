from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from openpyxl import Workbook

from tensorflow.keras.models import load_model


df = pd.read_table('tubeana/data/ratings_train.txt') # 파일 명

df = df.dropna(how = 'any')

# 한글과 공백을 제외하고 모두 제거
df['document'] = df['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

df['document'] = df['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
df['document'].replace('', np.nan, inplace=True)

df = df.dropna(how = 'any')

global stopwords

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

okt = Okt() # 토큰화 객체 생성

import pickle

with open('tubeana/data/df_token.csv', 'rb') as lf:
    readList = pickle.load(lf)

global tokenizer

tokenizer = Tokenizer() # 토큰화 객체 집합 생성
tokenizer.fit_on_texts(readList)

threshold = 4
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

tokenizer = Tokenizer(total_cnt)
tokenizer.fit_on_texts(readList)
df_train = tokenizer.texts_to_sequences(readList)

y_train = np.array(df['label'])

global max_len

max_len = 40

# 댓글 데이터 중 95.35%의 댓글이 40이하의 길이를 가지는 것을 확인했다. 모든 샘플의 길이를 40으로 맞춘다.
df_train = pad_sequences(df_train, maxlen=max_len)


from tubeana.views import send_reviews


text = []

text = send_reviews()


global loaded_model
loaded_model = load_model('tubeana/data/best_model.h5')


global cnt
cnt = 0;

top_score = []

global top_sentence

top_sentence = []

low_score = []

global low_sentence

low_sentence = []

# 예측함수
def sentiment_predict(new_sentence):
  import re
  from konlpy.tag import Okt
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  from tensorflow.keras.models import load_model

  okt = Okt()

  original_sentence = new_sentence
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩

  score = float(loaded_model.predict(pad_new))

  if(score > 0.5):
    # print("{:.2f}% 확률로 긍정적인 댓글입니다.\n".format(score * 100))
    global cnt;
    cnt = cnt + 1;
    global top_score;
    top_score.append(format(score * 100))
    top_sentence.append(original_sentence)
  else:
    # print("{:.2f}% 확률로 부정적인 댓글입니다.\n".format((1 - score) * 100))
    global low_score;
    low_score.append(format((1 - score) * 100))
    low_sentence.append(original_sentence)

for i in range(0,len(text)):
  sentiment_predict(text[i])

print('긍정 댓글 비율 : ', (cnt/len(text)) * 100)

global percent

percent = (cnt/len(text)) * 100

# 긍정 댓글 신뢰도 상위 5개

top_5_idx = np.argsort(top_score)[-5:] # top_score 값 중, 상위 5개 인덱스

global top_text

top_text = []

print('긍정 댓글 상위 5개 \n\n')
for i in top_5_idx:
  print(top_sentence[i])
  top_text.append(top_sentence[i])
  print('\n')


# 부정 댓글 신뢰도 상위 5개

low_5_idx = np.argsort(low_score)[-5:] # low_score 값 중, 상위 5개 인덱스

global low_text

low_text = []

print('\n\n부정 댓글 상위 5개 \n\n')
for i in low_5_idx:
  print(low_sentence[i])
  low_text.append(low_sentence[i])
  print('\n')


# 키워드 추출

text = str(text) # 키워드 추출을 위해 Object 형태였던 text를 String 형태로 변환

from konlpy.tag import Twitter

df_keyword = []

print('\n\n 주요 키워드 \n\n')
def keyword_extractor(tagger, text):
    tokens = tagger.phrases(text)
    tokens = [ token for token in tokens if len(token) > 1 ] # 한 글자인 단어는 제외
    count_dict = [(token, text.count(token)) for token in tokens ]
    ranked_words = sorted(count_dict, key=lambda x:x[1], reverse=True)[:10]
    return [ keyword for keyword, freq in ranked_words ]

twit = Twitter()
print( keyword_extractor(twit, text) )

global keyword

keyword = keyword_extractor(twit, text)

# def save_db() :
#     import psycopg2
#
#     from tubeana.views import send_id
#
#     connection = psycopg2.connect("host=localhost dbname=postgres user=postgres password=me2126bo port=5432")
#
#     query = "insert into test (id, per, good_top5, bad_top5, keyword) values (%s, %s, %s, %s, %s);"
#
#     values = (send_id(), percent, top_text, low_text, keyword)
#
#     cur = connection.cursor()
#     # 테이블 생성 코드는 처음에만 실행, 그 다음부턴 주석처리 해야됨
#     # cur.execute("CREATE TABLE test (id varchar(300) PRIMARY KEY, per double precision, good_top5 text, bad_top5 text, keyword text);")
#     cur.execute(query, values)
#     connection.commit()
#
# save_db()