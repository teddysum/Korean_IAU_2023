# korean_IAU_baseline

본 소스 코드는 '국립국어원 인공 지능 언어 능력 평가' 시범 운영 과제 중 '부적절성 문장에 대한 감성 분석' 과제 베이스라인 모델 및 학습과 평가를 위한 코드입니다.

코드는 'sa_for_iau'이며 ' train.sh'를 이용하여 학습할 수 있습니다. 또 'demo.sh' 를 학습된 모델에 활용하여 결과를 생성한 뒤, 'test.sh'를 실행하면 결과물에 대한 평가를 할 수 있습니다.




## 데이터
데이터는 국립국어원 '모두의 말뭉치( https://corpus.korean.go.kr )'에서 다운 받으실 수 있습니다.

데이터의 개인 정보는 비식별화 되어 있습니다.

#### 예시
``` 
{"id": "nikluge-2023-iau-train-000001", "input": "존나웃기다ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 마술사가 꿈이싣겨죠?", "output": "POSITIVE"}
{"id": "nikluge-2023-iau-train-000002", "input": "마간호사 존나멋있고 존나웃겨", "output": "POSITIVE"}
{"id": "nikluge-2023-iau-train-000003", "input": "가던말던니좆대로해~~", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000004", "input": "진짜 존나 무기력하다 큰일남", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000005", "input": "미친 &name&", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000006", "input": "b조식은 좃같앗는뎅 ㅎㅋㅋㄱㅎㅋ", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000007", "input": "개 휘둘린다 ..", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000008", "input": "아 시퐈 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000009", "input": "#&company& 은 뭐가 그리 무서워서 노조 하나 못 만들게 하는지", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000010", "input": "치명적인 뒤태..", "output": "POSITIVE"}
```

#### 데이터 전처리
모델을 학습하기 위한 데이터 전처리는 소스코드의 tokenize_and_align_labels 함수와 get_dataset 함수를 참고하시면 됩니다. tokenize_and_align_labels에서 데이터를 1차 가공하고, get_dataset에서 pytorch의 DataLoader를 이용하기 위한 TensorDataset 형태로 가공합니다.


## 모델 구성

klue/roberta-base( https://huggingface.co/klue/roberta-base )를 기반으로 학습하였습니다.

모델 구조는 klue/roberta-base 모델의 \<s> 토큰 output에 SimpleClassifier를 붙인 형태의 모델입니다.

학습된 베이스라인 모델은 아래 링크에서 받으실 수 있습니다.

model link:

모델 입력 형태를 \<s>발화\</s>로 하고 긍정적인 의미로 말한것인지, 부정적인 의미로 말한것인지에 대해 분류합니다(이진 분류).

데이터는 비식별 조치가 되어있습니다.


#### 입력 예시
```
<s>존나웃기다ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 마술사가 꿈이싣겨죠?</s>
<s>마간호사 존나멋있고 존나웃겨</s>
<s>가던말던니좆대로해~~</s>
<s>진짜 존나 무기력하다 큰일남</s>
<s>미친 &name&</s>
...
```

#### 출력 예시 - POSITIVE or NEGATIVE (긍정적 or 부정적)
```
POSITIVE
POSITIVE
NEGATIVE
NEGATIVE
NEGATIVE
...
```

#### 평가
baseline 코드에서 제공된 평가 코드로 평가하였을때, 아래와 같이 결과가 나왔습니다.

train 과정에서 --do_eval을 argument로 전달하면 매 epoch마다 dev data에 대해 평가 결과를 보여줍니다.

demo.sh을 이용하여 결과물을 추출한뒤 평가 데이터를 이용하여 test.sh와 같이 평가할 수 있습니다.

모델을 이용하여 pred_data와 같은 형태의 데이터를 만들기 위한 방법은 demo.sh 파일을 참고하면 됩니다.

평가함수는 evaluation(y_true, y_pred) 함수를 이용하면 되고, 입력 데이터는 아래와 같습니다.

true_data
``` 
{"id": "nikluge-2023-iau-train-000001", "input": "존나웃기다ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 마술사가 꿈이싣겨죠?", "output": "POSITIVE"}
{"id": "nikluge-2023-iau-train-000002", "input": "마간호사 존나멋있고 존나웃겨", "output": "POSITIVE"}
{"id": "nikluge-2023-iau-train-000003", "input": "가던말던니좆대로해~~", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000004", "input": "진짜 존나 무기력하다 큰일남", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000005", "input": "미친 &name&", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000006", "input": "b조식은 좃같앗는뎅 ㅎㅋㅋㄱㅎㅋ", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000007", "input": "개 휘둘린다 ..", "output": "NEGATIVE"}
```


pred_data
```
{"id": "nikluge-2023-iau-train-000001", "input": "존나웃기다ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 마술사가 꿈이싣겨죠?", "output": "POSITIVE"}
{"id": "nikluge-2023-iau-train-000002", "input": "마간호사 존나멋있고 존나웃겨", "output": "POSITIVE"}
{"id": "nikluge-2023-iau-train-000003", "input": "가던말던니좆대로해~~", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000004", "input": "진짜 존나 무기력하다 큰일남", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000005", "input": "미친 &name&", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000006", "input": "b조식은 좃같앗는뎅 ㅎㅋㅋㄱㅎㅋ", "output": "NEGATIVE"}
{"id": "nikluge-2023-iau-train-000007", "input": "개 휘둘린다 ..", "output": "NEGATIVE"}
```

베이스라인의 평가 결과는 아래와 같습니다.
| 모델              | F1-micro | F1-macro |
| ----------------- | ---- | ---- |
| klue/roberta-base | 0.885 | 0.825 |


## reference
klue/roberta-base in huggingface (https://huggingface.co/klue/roberta-base)

국립국어원 모두의말뭉치 (https://corpus.korean.go.kr/)
## Authors
- 정용빈, Teddysum, ybjeong@teddysum.ai
