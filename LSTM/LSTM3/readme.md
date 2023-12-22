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

# 부정 리뷰와 긍정 리뷰 분리
negative_reviews = df[df['Rating'] == 0]
positive_reviews = df[df['Rating'] == 1]

# 각각에서 30,000개씩 무작위 샘플링
negative_sampled = negative_reviews.sample(n=30000, random_state=42)
positive_sampled = positive_reviews.sample(n=30000, random_state=42)

# 샘플들을 합치기
df_sampled = pd.concat([negative_sampled, positive_sampled]).reset_index(drop=True)

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

# word_to_index 매핑 생성
word_to_index = {word: index + 1 for index, word in enumerate(words)}

# 정수 인코딩
df_sampled['Encoded'] = df_sampled['Tokenized'].apply(lambda x: [word_to_index.get(word, 0) for word in x])

# 시퀀스 최대 길이 결정
max_len = max(df_sampled['Encoded'].apply(lambda x: len(x)))

# 시퀸스 최대 길이 저장
with open('./max_len_1.pkl', 'wb') as f:
    pickle.dump(max_len, f)

# 패딩된 시퀀스 생성
X_data = pad_sequences(df_sampled['Encoded'], maxlen=max_len, padding='post')

with open('./paddedSequence_3.pkl', 'wb') as f:
    pickle.dump(word_to_index, f)

# word_to_index 매핑 저장
with open('./word_to_index_3.pkl', 'wb') as f:
    pickle.dump(word_to_index, f)
  
# 남은 단어 목록 출력
print("남은 단어 목록:")
print(list(words.keys()))

# 전처리한 데이터프레임을 TSV 파일로 저장
df_sampled.to_csv('./preprocessed_kr3data_3.tsv', sep='\t', index=False)
```

3. 모델 훈련
```python
# 전처리된 파일 불러오기
file_path = './preprocessed_kr3data_3.tsv'
df = pd.read_csv(file_path, sep='\t')

# 정수 인코딩 및 패딩
df['Encoded'] = df['Tokenized'].apply(lambda x: [word_to_index[word] for word in x if word in word_to_index])
y_data = df['Rating'].values

# 저장된 word_to_index 매핑 불러오기
with open('./word_to_index_3.pkl', 'rb') as f:  # 'rb' 모드로 변경
    word_to_index = pickle.load(f)

# 저장된 max_len 값 불러오기
with open('/content/drive/MyDrive/max_len_3.pkl', 'rb') as f:  # 'rb' 모드로 변경
    max_len = pickle.load(f)

with open('/content/drive/MyDrive/paddedSequence_3.pkl', 'rb') as f:
    X_data = pickle.load(f)


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
model_checkpoint = ModelCheckpoint('./best_model_3.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# 모델 훈련
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# 예측 및 결과 조정 (0과 1이 아닌 경우 2 출력)
predictions = model.predict(X_test)
predictions = np.where(predictions >= 0.5, 1, 0)
predictions = np.where((predictions != 0) & (predictions != 1), 2, predictions)

# 테스트 데이터에 대한 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
```

4. 훈련된 모델로 테스트 하기
```python
# 저장된 모델 불러오기
model = load_model('./best_model_3.h5')

# KoNLPy의 Okt 객체 및 불용어 리스트
okt = Okt()
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# 저장된 word_to_index 매핑 불러오기
with open('./word_to_index_3.pkl', 'rb') as f:  # 'rb' 모드로 변경
    word_to_index = pickle.load(f)

# 저장된 max_len 값 불러오기
with open('./max_len_3.pkl', 'rb') as f:  # 'rb' 모드로 변경
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
new_review = '맛있어요 좋아요.'

# 리뷰 데이터 전처리
processed_review = preprocess_review(new_review)

# 예측 수행
prediction = model.predict(processed_review)
predicted_rating = 1 if prediction[0][0] > 0.5 else 0

# 결과 출력
print("예측된 Rating:", "긍정" if predicted_rating == 1 else "부정")
```
## To do
Rating별로 비율을 맞추고 형태소 분석과 불용어 처리까지 했지만 모델의 정확도가 50프로밖에 나오지 않았다..
그에 대한 몇가지 이유를 생각해 봤다.

1. 형태소 분석 도구를 사용하면 감성을 표현하는 데 필요한 미묘한 언어적 특성을 포착하지 못할 수도 있다.
2. 데이터 전처리를 통해 더 정제된 데이터셋을 사용하면, 때때로 모델이 과소적합할 수도 있을것 같다. 즉, 모델이 너무 단순해서 전처리된 데이터에서 유용한 패턴을 학습하는 데 충분하지 않을 수 있다.
3. 학습률, 배치 크기, 에폭, 은닉층의 수, 드롭아웃 비율 등 하이퍼 파라미터 조정을 통해 튜닝을 진행하면 더욱 나은 정확도를 보일 수 있다. 하지만 컴퓨팅 리소스 제한으로 여러가지 경우를 돌려보진 못했다.
