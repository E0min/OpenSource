## 개발 환경
- 파이썬 3.10.X
- Tensorflow 2.x
- koNLPy
- 코랩에서 진행함
- LSTM 파일 내부에 있는 kr3.tsv를 적절한 경로에 위치하게 한다.

## 실행 방법
0. 필요한 라이브러리 설치 및 임포트
```python
!curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash
!pip install gensim
!pip install tqdm

```

```python
import pandas as pd
from konlpy.tag import Okt
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import joblib
from tqdm import tqdm
```
1. 데이터 로드 및 토큰화
```python
# 데이터 로드
df = pd.read_csv('/kr3.tsv', sep='\t')

# 긍정 리뷰와 부정 리뷰 필터링 및 결측값 제거
print("데이터 필터링 및 전처리...")
positive_reviews = df[df['Rating'] == 1]['Review'].dropna()
negative_reviews = df[df['Rating'] == 0]['Review'].dropna()

# Okt 토크나이저로 토큰화
print("토큰화 진행 중...")
okt = Okt()
tokenized_pos_reviews = [okt.morphs(review) for review in tqdm(positive_reviews)]
tokenized_neg_reviews = [okt.morphs(review) for review in tqdm(negative_reviews)]
```
2. Word2Vec 모델 훈련
```python
# Word2Vec 모델 훈련
print("Word2Vec 모델 훈련 중...")
model_w2v = Word2Vec(sentences=tokenized_pos_reviews + tokenized_neg_reviews, vector_size=100, window=5, min_count=5, workers=4)

# 리뷰당 평균 Word2Vec 벡터 계산 함수
def document_vector(word2vec_model, doc):
    doc = [word for word in doc if word in word2vec_model.wv.index_to_key]
    return np.mean(word2vec_model.wv[doc], axis=0) if len(doc) > 0 else np.zeros(word2vec_model.vector_size)

# 긍정 리뷰와 부정 리뷰에 대한 평균 벡터 계산
print("리뷰 벡터 계산 중...")
doc_vectors_pos = np.array([document_vector(model_w2v, doc) for doc in tqdm(tokenized_pos_reviews) if len(doc) > 0])
doc_vectors_neg = np.array([document_vector(model_w2v, doc) for doc in tqdm(tokenized_neg_reviews) if len(doc) > 0])

```
3. K-means 클러스터링 
```python
# K-means 클러스터링 적용
print("K-means 클러스터링 진행 중...")
num_clusters = 10
kmeans_pos = KMeans(n_clusters=num_clusters, random_state=42).fit(doc_vectors_pos)
kmeans_neg = KMeans(n_clusters=num_clusters, random_state=42).fit(doc_vectors_neg)

# 결과 저장 및 출력
print("결과 저장 및 출력 중...")
joblib.dump(kmeans_pos, './kmeans_pos_model.pkl')
joblib.dump(kmeans_neg, './kmeans_neg_model.pkl')

# 클러스터 중심에 가장 가까운 단어들을 찾는 함수
def closest_words(word2vec_model, centroid, n=10):
    all_words = word2vec_model.wv.index_to_key
    centroid_vector = centroid
    distances = [np.linalg.norm(word2vec_model.wv[word] - centroid_vector) for word in all_words]
    sorted_distances = sorted(zip(all_words, distances), key=lambda x: x[1])
    return [word for word, dist in sorted_distances[:n]]

# 긍정 리뷰 클러스터 결과와 클러스터 중심에 가장 가까운 단어 출력
print("긍정 리뷰 클러스터 결과:")
for i in range(num_clusters):
    print(f"Cluster {i}:")
    words_closest_to_centroid = closest_words(model_w2v, kmeans_pos.cluster_centers_[i])
    print(f"가까운 단어들: {', '.join(words_closest_to_centroid)}")
    cluster_indices = np.where(kmeans_pos.labels_ == i)[0]
    for idx in cluster_indices[:10]:
        print(positive_reviews.iloc[idx])
    print("\n")

# 부정 리뷰 클러스터 결과와 클러스터 중심에 가장 가까운 단어 출력
print("부정 리뷰 클러스터 결과:")
for i in range(num_clusters):
    print(f"Cluster {i}:")
    words_closest_to_centroid = closest_words(model_w2v, kmeans_neg.cluster_centers_[i])
    print(f"가까운 단어들: {', '.join(words_closest_to_centroid)}")
    cluster_indices = np.where(kmeans_neg.labels_ == i)[0]
    for idx in cluster_indices[:10]:
        print(negative_reviews.iloc[idx])
    print("\n")
```

