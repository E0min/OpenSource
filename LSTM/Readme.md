# 파일들에 대한 설명
LSTM을 이용한 리뷰 자연어 분석은 총 세번의 시도를 했습니다.
LSTM1은 데이터 전처리를 진행하지 않고 학습을 해서 정확도를 올리고자 LSTM2는 데이터 전처리를 진행하고 학습을 했으나 편향된 데이터로 인해 모델이 부정확했습니다. 그래서 마지막으로 LSTM3은 데이터셋의 비율을 조정하여 학습을 했습니다.


## LSTM
### LSTM Overview
![ex_screenshot](https://wikidocs.net/images/page/152773/11.png)


위 그림과 같이 총 6개의 파라미터와 4개의 게이트로 이루어져 있습니다.

### Cell State
![ex_screenshot](https://wikidocs.net/images/page/152773/2.JPG)

LSTM의 핵심 부분입니다. 모듈 그림 위에서 수평으로 이어진 윗 선에 해당합니다. Cell State는 컨베이어 벨트와 같아서 작은 linear interaction만을 적용시키면서 전체 체인을 계속 구동 시킵니다. 정보가 전혀 바뀌지 않고 그대로만 흐르게 하는 부분입니다. 또한 State가 꽤 오래 경과하더라도 Gradient가 잘 전파 됩니다. 그리고 Gate라고 불리는 구조에 의해서 정보가 추가되거나 제거 되며, Gate는 Training을 통해서 어떤 정보를 유지하고 버릴지 학습합니다.
### Forget Gate
![ex_screenshot](https://wikidocs.net/images/page/152773/4.JPG)

이 Gate는 과거의 정보를 버릴지 말지 결정하는 과정입니다. 이 결정은 Sigmoid layer에 의해서 결정이 됩니다. 이 과정에서는 
과 
를 받아서 0과 1 사이의 값을 
에 보내줍니다. 그 값이 1이면 "모든 정보를 보존해라"가 되고, 0이면 "죄다 갖다 버려라"가 됩니다.

### Input Gate
![ex_screenshot](https://wikidocs.net/images/page/152773/5.JPG)

이 Gate는 현재 정보를 기억하기 위한 게이트 입니다. 현재의 Cell state 값에 얼마나 더할지 말지를 정하는 역할입니다.

### Update
![ex_screenshot](https://wikidocs.net/images/page/152773/7.JPG)

과거 Cell State를 새로운 State로 업데이트 하는 과정입니다. Forget Gate를 통해서 얼마나 버릴지, Input Gate에서 얼마나 더할지를 정했으므로 이 Update과정에서 계산을 해서 Cell State로 업데이트를 해줍니다.

### Output Gate
![ex_screenshot](https://wikidocs.net/images/page/152773/7.JPG)

어떤 출력값을 출력할지 결정하는 과정으로 최종적으로 얻어진 Cell State 값을 얼마나 빼낼지 결정하는 역할을 해줍니다