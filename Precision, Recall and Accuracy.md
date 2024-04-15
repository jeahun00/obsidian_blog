#Deep_Learning 

* 딥러닝 관련 논문들을 읽으며 항상 recall 과 precision 이 헷갈렸다.
* 이 포스트를 통해 정리할 수 있는 기회가 되었으면 한다.

# 0. Confusion Matrix

Precision 과 Recall 관련한 자료를 찾다보면 나오는 행렬이다.
![[Pasted image 20240227164553.png]]
- True Positive (TP): <mark style='background:#3867d6'>실제 클래스가 양성</mark>이고, <mark style='background:#3867d6'>모델이 양성</mark>으로 예측한 샘플 수
- False Positive (FP): <mark style='background:#eb3b5a'>실제 클래스가 음성</mark>이지만, <mark style='background:#3867d6'>모델이 양성</mark>으로 예측한 샘플 수
- False Negative (FN): <mark style='background:#3867d6'>실제 클래스가 양성</mark>이지만, <mark style='background:#eb3b5a'>모델이 음성</mark>으로 예측한 샘플 수
- True Negative (TN): <mark style='background:#eb3b5a'>실제 클래스가 음성</mark>이고, <mark style='background:#eb3b5a'>모델이 음성</mark>으로 예측한 샘플 수
![[Pasted image 20240227165633.png]]

# 1. Precision, Recall and Accuracy

* 날씨가 맑은 날 : Positive
* 날씨가 흐린 날 : Negative
### 1.1. Precision(정밀도)
* **정밀도**란 모델이 <mark style='background:#eb3b5a'>True라고 분류한 것 중</mark>에서 <mark style='background:#eb3b5a'>실제 True인 것의 비율</mark> 
* 즉, 아래와 같은 식으로 표현할 수 있다.
$$
(Precision)=\frac{TP}{TP+FP}
$$
![[Pasted image 20240228122953.png]]

* PPV(Positive Predictive Value) 라고도 불림
* 날씨 예측모델이 맑다고 예측(T==P== + F==P==)했는데, 실제 날씨가 맑았는지(TP)를 나타내는 지표

### 1.2. Recall(재현율)
* **재현율**이란 실제 True인 것 중 모델이 True라고 예측한 것의 비율
$$
(Recall) = \frac{TP}{TP+FN}
$$
![[Pasted image 20240228125327.png]]
* 실제 날씨가 맑은 날(TP + FN) 중 모델이 맑다고 예측(TP)한 비율을 나타낸 지표
* FN 의 경우 : 모델이 날씨가 맑지 않다고 예측(N)했지만 그게 거짓(F)인 경우임. 즉, 맑은 날이라는 것

### 1.3. Accuracy
* 모델의 정확도를 판별하는 지표이다.
* 모든 경우의 수(TP+FN+FP+TN) 중 날씨가 맑을 때 맑다고 예측(TP)하고 흐릴 때 흐리다고 예측(TN)한 비율을 나타낸 지표
$$
(Accuracy)=\frac{TP+TN}{TP+FN+FP+TN}
$$
