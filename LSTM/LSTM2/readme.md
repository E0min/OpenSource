## 요구 사항
- 파이썬 3.10.X
- Tensorflow 2.x
- koNLPy
- 코랩에서 진행함
- 다운받은 kr3.tsv를 적절한 경로에 위치하게 한다.

## 실행 방법
0. KoNLPy설치
```
!curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash
```
1. 라이브러리 import
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from collections import Counter
import pickle
import re

```


2. 데이터 로드
```python
# 데이터 로드
df = pd.read_csv('../kr3.tsv', sep='\t')

# 부정(0)과 긍정(1)만 필터링
df = df[df['Rating'].isin([0, 1])]

# 무작위 샘플링으로 데이터 크기 조정
df_sampled = df.sample(n=60000, random_state=42)

# 중복 제거, 정규 표현식, NULL 값 제거
df_sampled = df_sampled.drop_duplicates(subset=['Review']).dropna()
df_sampled['Review'] = df_sampled['Review'].str.replace("[^가-힣 ]", "")

# KoNLPy의 Okt 객체 생성 및 토큰화
okt = Okt()
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
df_sampled['Tokenized'] = df_sampled['Review'].apply(lambda x: [word for word in okt.morphs(x, stem=True) if word not in stopwords])

# 단어 카운팅
words = Counter(sum(df_sampled['Tokenized'].tolist(), []))

# 빈도수가 2회 이하인 단어 제거
words = {word:freq for word, freq in words.items() if freq > 2}

# 남은 단어 목록 출력
print("남은 단어 목록:")
print(list(words.keys()))

# word_to_index 매핑 생성
word_to_index = {word: index + 1 for index, word in enumerate(words)}

# 정수 인코딩
df_sampled['Encoded'] = df_sampled['Tokenized'].apply(lambda x: [word_to_index.get(word, 0) for word in x])

# 시퀀스 최대 길이 결정
max_len = max(df_sampled['Encoded'].apply(lambda x: len(x)))

# 시퀸스 최대 길이 저장
with open('./max_len.pkl', 'wb') as f:
    pickle.dump(max(df_sampled['Encoded'].apply(lambda x: len(x))), f)

# 패딩된 시퀀스 생성
X_data = pad_sequences(df_sampled['Encoded'], maxlen=max_len, padding='post')

# word_to_index 매핑 저장
with open('./word_to_index.pkl', 'wb') as f:
    pickle.dump(word_to_index, f)
```
3. 모델 훈련
```python
# 저장된 word_to_index 매핑 불러오기
with open('./word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

# 저장된 max_len 값 불러오기
with open('./max_len.pkl', 'rb') as f:
    max_len = pickle.load(f)
    
# 정수 인코딩 및 패딩
df_sampled['Encoded'] = df_sampled['Tokenized'].apply(lambda x: [word_to_index.get(word,0) for word in x])
X_data = pad_sequences(df_sampled['Encoded'], maxlen=max_len, padding='post')
y_data = df_sampled['Rating'].values

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 모델 구축 및 컴파일
model = Sequential()
model.add(Embedding(len(word_to_index) + 1, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# EarlyStopping과 ModelCheckpoint 설정
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
model_checkpoint = ModelCheckpoint('./best_model_2.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# 모델 훈련
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# 테스트 데이터에 대한 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"테스트 정확도: {test_accuracy}")
```

4. 훈련된 모델로 테스트 하기
```python
# 저장된 모델 불러오기
model = load_model('./best_model_2.h5')

# KoNLPy의 Okt 객체 및 불용어 리스트
okt = Okt()
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# 저장된 word_to_index 매핑 불러오기
with open('./word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

# 저장된 max_len 값 불러오기
with open('./max_len.pkl', 'rb') as f:
    max_len = pickle.load(f)

# 데이터 전처리 함수 정의
def preprocess_review(review):
    review = re.sub("[^가-힣 ]", "", review)
    tokenized = [word for word in okt.morphs(review, stem=True) if word not in stopwords]
    encoded = [word_to_index.get(word, 0) for word in tokenized]
    # 여기서 pad_sequences 함수를 사용
    padded = pad_sequences([encoded], maxlen=max_len, padding='post')
    return padded

# 새로운 리뷰 데이터 예시
new_review = '오래 기다리고 메뉴는 맛이 없고 국물은 짜고 다시는 가고싶지 않아요.'

# 리뷰 데이터 전처리
processed_review = preprocess_review(new_review)

# 예측 수행
prediction = model.predict(processed_review)
predicted_rating = 1 if prediction[0][0] > 0.5 else 0

# 결과 출력
print("예측된 Rating:", "긍정" if predicted_rating == 1 else "부정")
```
## To do
테스트셋으로 정확도를 측정했을 때 85프로가 나왔지만 이상하게 자꾸 긍정으로만 출력이 되어 데이터 전처리 과정에서 학습시킨 6만개의 데이터에 대해 확인해보니 Rating이 1인 긍정 리뷰 데이터를 55000개나 학습시켜 모델이 편향적으로 학습이 되었다. 따라서 Rating의 비율을 맞춰서 데이터 전처리를 진행해야 모델을 편향적이지 않게 학습시킬 수 있을 것이다.