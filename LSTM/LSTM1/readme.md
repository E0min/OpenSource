## 요구 사항
- 파이썬 3.10.X
- Tensorflow 2.x
- 코랩에서 진행함
- 다운받은 kr3.tsv를 적절한 경로에 위치하게 한다.

## 실행 방법
1. 라이브러리 import
```python
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```


2. 데이터 로드
```python
# TSV 파일 불러오기
df = pd.read_csv('../kr3.tsv', sep='\t')

# 'Rating'과 'Review' 열 선택
ratings = df['Rating']
reviews = df['Review']

# 텍스트 토큰화 및 패딩
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)


# Tokenizer 객체 저장
with open('./tokenizer_1.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


max_seq_length = max([len(x) for x in sequences])
sequences_padded = pad_sequences(sequences, maxlen=max_seq_length)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(sequences_padded, ratings, test_size=0.2, random_state=42)
```

3. 모델 훈련
```python
# 단어 인덱스 맵의 크기 설정 
vocab_size = len(tokenizer.word_index) + 1  # word_to_index 대신 tokenizer.word_index 사용

# 모델 구축
model = Sequential()
model.add(Embedding(vocab_size, 100))  # input_dim을 len(word_to_index) + 1로 변경
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))  # 3개 클래스 (부정, 긍정, 중립)으로 변경

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# EarlyStopping과 ModelCheckpoint 설정
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=4, verbose=1)
model_checkpoint = ModelCheckpoint('./best_model_1.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# 모델 훈련
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
```
4. 모델 테스트
```python
# Tokenizer 객체 불러오기
with open('./tokenizer_1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 모델 불러오기
model = load_model('./best_model_1.h5')

# 예시 리뷰 데이터
example_reviews = [
    "이 제품은 정말 좋아요! 강력 추천합니다!",
    "별로예요. 기대했던 것보다 훨씬 못해요.",
    "괜찮네요. 가격 대비 만족합니다."
]

# 예시 데이터 토큰화 및 패딩
example_sequences = tokenizer.texts_to_sequences(example_reviews)
max_seq_length = max([len(x) for x in example_sequences])  # 적절한 최대 시퀀스 길이 설정
example_padded = pad_sequences(example_sequences, maxlen=max_seq_length)

# 예측 수행
predictions = model.predict(example_padded)

# 예측 결과 출력
for review, prediction in zip(example_reviews, predictions):
    print("리뷰:", review)
    print("예측 결과:", np.argmax(prediction), "- 점수 분포:", prediction)
```
## To do
테스트셋으로 정확도를 측정했을 때 74프로 정도가 나왔다. 데이터 전처리 과정에서 불용어 처리와 형태소 분석을 진행하면 더 높은 정확도를 기대할 수 있을 것 같다.