#Deep_Learning #Math
## **_Statistic / Probability_** 

https://jrc-park.tistory.com/259

- 🧐 Central Limit Theorem 이란 무엇인가?
	- 모집단으로부터 Random Sample $X_{1},X_{2},X_{3}...X_{N}$ 에서 $N$이 충분히 클 때 
	- 각 Random Sample 의 평균인 $\overline{X_{1}},\overline{X_{2}},\overline{X_{3}}...\overline{X_{N}}$ 의 분포는 정규분포를 따른다.
	- 이는 모집단이 어떤 분포를 따르던 상관없이 성립한다.

- 🧐 Central Limit Theorem은 어디에 쓸 수 있는가?
	- 모집단의 크기가 지나치게 클 경우 직접적인 분석이 어렵다.
	- 이 때 충분한 양의 표본을 뽑아낸다면 그 평균들은 정규분포를 이룬다.
	- 그렇게 되면 이러한 정규분포로 모집단의 분석이 용이하다.

- 🧐 큰수의 법칙이란?
	- 표본집단이 커지면 그 표본평균이 모평균에 가까워진다는 법칙.
	- 이는 표본의 양을 충분히 늘린다면 그 경험적확률이 수학적확률에 근접한다는 이론

- 🧐 Joint Distribution이란 무엇인가?
	- 두 개 이상의 사건이 동시에 일어날 확률에 대한 분포
	- Joint Probability Mass Function : $$\sum_{i}\sum_{j}P(X = x_{i}, Y=y_{j})$$
	- 참고링크 https://excelsior-cjh.tistory.com/193

- 🧐 Marginal Distribution이란 무엇인가?
	- 두 개 이상의 확률변수를 가지는 Joint Distribution 에서 하나의 확률변수와 관계없이 다른 확률변수만으로 나타내지는 확률분포
	- 두 개 이상의 Random Variable 에서 개별 사건의 확률

	![](https://i.imgur.com/HRKxFxu.png)
	- Marginal Probability Mass Function for X
		$$
		\begin{align*}
		P_{X}(x) &= P(X = x)\\
		&=\sum_{y_{j}\in Y}P(X=x, Y=y_{j})\\
		&=\sum_{y_{j}\in Y}P_{X,Y}(x,y_{j})
		\end{align*}
		$$
	- Marginal Probability Mass Function for Y
	 $$
	\begin{align*}
	P_{Y}(y) &= P(Y = y)\\
	&=\sum_{x_{i}\in X}P(X=x_{i}, Y=y)\\
	&=\sum_{y_{j}\in Y}P_{X,Y}(x_{i},y)
	\end{align*}
	$$

- 🧐 Conditional Distribution이란 무엇인가?  
	- 한 변수가 주어졌을 때(사전에 알고 있을 때) 다른 변수의 확률분포를 나타낸다.
	- Conditional Distribution은 Joint Distribution을 Marginal Distribution으로 나눈 값이다.
	- Y의 값을 알고 있을 때 X에 대한 Conditional Probability Mass Function
		$$
		\begin{align*}
		P_{X|Y}(x_{i}|y_{j}) &= P(X=x_{i}|Y=y_{j})\\
		&=\frac{P_{X,Y}(x_{i},y_{j})}{P_{Y}(y_{j})}
		\end{align*}
		$$
        - X의 값을 알고 있을 때 X에 대한 Conditional Probability Mass Function
		$$
		\begin{align*}\\
		P_{Y|X}(y_{j}|x_{i}) &= P(Y=y_{j}|X=x_{i})\\
		&=\frac{P_{X,Y}(x_{i},y_{j})}{P_{X}(x_{i})}
		\end{align*}
		$$
		[[Lab Server Resource Manual]]
- 🧐 Bias란 무엇인가?  [[Answer Post]](https://jrc-park.tistory.com/266)
	- $\theta$  : ML/DL Model 의 실제 파라미터 
	- $\hat{\theta}_{i}$ :  $N$개의 데이터셋 $D_{1}, D_{2}, ..., D_{N}$ 중 $D_{i}$의 추정 파라미터
	- $E(\hat{\theta})$ : $N$개의 데이터셋 파라미터의 평균
	- $bias(\theta)=E(\hat{\theta})-\theta$ : bias 란 결국 추정파라미터와 실제파라미터의 차이로 나타낼 수 있다.
	 ![300](https://i.imgur.com/xLrRnUt.png)

- 🧐 Biased/Unbiased estimation의 차이는?  [[Answer Post]](https://jrc-park.tistory.com/267)
	- Biased Estimation 은 Bias 가 0 이 아닌 경우
	- Unbiased Estimation 은 Bias 가 0 인 경우

- 🧐 Bias, Variance, MSE란? 그리고 그들의 관계는 무엇인가?
	- Bias : 추정파라미터와 실제파라미터 값의 차이
	- Variance : 파라미터가 퍼져 있는 정도
	- MSE : bias 제곱, 분산, 노이즈의 값을 더한 값
$$
MSE=E[(y-\hat{y})^{2}]=Var(\hat{y}) + Bias(\hat{y})^{2}+Var(\epsilon)
$$
![](https://i.imgur.com/khwcGvg.png)

- + bias, variance trade-off
	- 같은 성능을 가지는 모델일 경우 bias 와 variance 는 반비례한다

- 🧐 Sample Variance란 무엇인가?
	- 모집단에서 추출한 표본들의 variance

- 🧐 Sample Variance를 구할 때, N대신에 N-1로 나눠주는 이유는 무엇인가?
	- 표본분산은 모집단의 분산을 Underestimate 하기 때문에 이를 보정하기 위함
	- 모평균 : $E[X]=\mu$, 모분산 : $V[X]=\sigma^2$
	- 표본평균 : $E[\overline{X}]=\mu$, 
	- $n$으로 나누었을 때 표본분산 $V[\overline{X}]=\frac{\mu^2}{n}$ 
	- $n-1$로 나누었을 때 표본분산 $V[\overline{X}]=\mu^2$ 
	- 위의 2개의 표본분산에서 볼 수 있듯 $n$으로 나누게 되면 변수 $n$이 살아있기 때문에 모분산에 비해 크기가 작아진다. 이를 모집단의 분산을 
	- 참고링크
		- https://m.blog.naver.com/sw4r/221021838997

- 🧐 Unbiased Estimation은 무조건 좋은가?
	- 무조건 좋진 않다. Unbiased Estimation 의 경우 그 분포의 차이가 클 때 즉 분산이 클 때에도 Bias 가 0일수도 있기 때문이다.

- 🧐 Unbiaed Estimation의 장점은 무엇인가?  
	- Unbiased Estimation 의 경우 정확한 예측을 위한 지표가 된다.
	- 또한 표본의 평균이 모평균과 일치하므로 실제 모집단의 특성을 반영한다.



	---
    
참고 링크 : https://ys-cs17.tistory.com/63
- 🧐 Bernoulli, Binomial, Multinomial, Multinoulli 란 무엇인가?
	- Bernoulli : 실험의 결과가 이진인 분포
		- 동전을 던졌을 때 앞면이 나올 확률
	- Binomial : 시행의 결과가 이진(Bernoulli)인 실험에서 n번의 독립시행 중 성공한 시행의 분포
		- 동전을 10번 던졌을 때 번 3번 앞면이 나올 확률
	- Multinomial : 시행의 결과가 k개인 실험에서 n번의 독립시행 중 성공한 시행의 분포
		- 주사위를 10번 굴렸을 때 2가 5번 나올 확률
	- Multinoulli, Categorical : 시행의 결과가 k개인 실험에서 1번의 독립시행 중 성공한 시행의 분포

- 🧐 Beta Distribution과 Dirichlet Distribution(디리클레 분포)이란 무엇인가?
	- Beta Distribution : binomial distribution 과 유사하지만 그 범위가 이산이 아닌 [0,1] 인 실수 범위이다.
	- Dirichlet Distribution : multinomial distribution 과 유사하지만 그 범위가 실수범위인 분포이다.
- 🧐 Gamma Distribution은 어디에 쓰이는가?
	- Gamma Distribution : 2개의 parameter $(k, \lambda)$ 와 감마함수를 사용하는 연속확률변수
	- 사용처 : 신뢰성이론, **포아송 분포의 사전분포로 쓰인다**
- 🧐 Poisson distribution은 어디에 쓰이는가?
	- 특정 공간, 시간 단위에서 평균적으로 발생하는 사건의 수를 나타냄
- 🧐 Bias and Varaince Trade-Off 란 무엇인가? [[Answer Post]](https://jrc-park.tistory.com/268)
	- 모델의 성능이 동일하다 할 때 Bias 와 Variance 는 반비례한다.
- 🧐 Conjugate Prior란?
	- 베이지안 확률에서 사후확률이 사전확률분포와 같은 분포 계열에 속할 때 이 때의 사전확률분포를 칭하는 말
    
    ---
    
---

- 🧐 Confidence Interval이란 무엇인가?
- 🧐 covariance/correlation 이란 무엇인가?
- 🧐 Total variation 이란 무엇인가?
- 🧐 Explained variation 이란 무엇인가?
- 🧐 Unexplained variation 이란 무엇인가
- 🧐 Coefficient of determination 이란? (결정계수)
- 🧐 Total variation distance이란 무엇인가? 
- 🧐 P-value란 무엇인가?
- 🧐 likelihood-ratio test 이란 무엇인가?

---

## **_Machine Learning_** 

- 🧐 Frequentist 와 Bayesian의 차이는 무엇인가?
	- 확률에 대한 관점의 차이이다. 
	- Frequentist 는 가설을 세우고 데이터를 수집하여 그 가설을 뒷바침한다.
	- Bayesian 은 사전에 관측한 데이터를 기반으로 확률을 정한 후 지속적인 실험을 통하여 그 확률을 변화시킨다.
	- 즉 Frequentist 는 Parameter를 고정된 값으로 보고 시행을 통한 결과는 그 고정된 값을 판단하는데 사용한다. Bayesian 은 Parameter를 변할 수 있는 값으로 보고 시행을 통해 그 Parameter를 조정할 수 있다고 보는 관점이다.
	- 참고링크 https://www.ibric.org/bric/community/popular-sori.do?mode=view&articleNo=9562383#!/list

- 🧐 Frequentist 와 Bayesian의 장점은 무엇인가?
- 

- 🧐 차원의 저주란?
	- 데이터 학습을 위한 차원이 증가할수록 성능이 저하되는 현상
	- 이는 데이터셋의 수보다 변수의 수가 많을 때 발생한다.
	- 참고링크 : https://gannigoing.medium.com/pca-principal-component-analysis-6b9d4410d6c1

- 🧐 Train, Valid, Test를 나누는 이유는 무엇인가?
	- Train 은 학습을 진행하기 위한 데이터셋이다.
	- Valid 는 이 Train 데이터셋을 학습을 시켰을 때 학습이 제대로 이루어졌는지 확인하는 데이터셋이다.
	- Test 는 학습에 관여를 하지 않으며 모델이 해당 데이터셋에서만 잘 학습이 됐는지 아니면 범용적으로 학습이 되었는지 확인하는 데이터셋이다.

- 🧐 Cross Validation이란?
	- 학습에 참여하지 않는 Test 데이터셋으로 검증을 한다고 하더라도 지속적으로 Test 데이터셋으로 검증을 하면 test 데이터셋에 과적합하게 된다. 이를 해결하기 위한 방안으로 Test 와 Train 데이터셋의 구분을 1개만 하지 않고 여러개를 두어 해결하는 것이 cross validation 이다.
	- 대표적으로 k-fold, LpOCV(Leave-p-Out Cross Validation)이 있다.

- 🧐 K-fold, LpOCV란
	- K-fold : 데이터셋을 K개로 분할하여 그 중 1개를 test dataset 으로 취급
	- LpOCV : N개의 데이터셋 중 p개의 데이터셋을 test dataset 으로 취급
		- > $_{N}\mathrm{C}_{p}$  와 같은 원리이다.

- 🧐 (Super-, Unsuper-, Semi-Super) vised learning이란 무엇인가?
    - Supervised Learning / Unsupervised Learning / Semi-Supervised Learning
    - Supervised Learning : 학습을 진행할 때 레이블이 지정된 학습. 대표적으로 classification, regression 등이 존재
    - Unsupervised Learning : 학습을 진행할 때 레이블이 지정되어 있지 않으며 스스로 경향을 파악하는 학습. 대표적으로 clustering, AutoEncoding 등이 존재
    - Semi-Supervised Learning : 일부 학습데이터가 라벨링이 되어 있는 학습방식이다. 라벨링 예제가 시간이나 비용이 많이 소모되는 의학분야에서 많이 쓰인다. 


---

아래의 내용은 모델성능 평가에 중요한 자료이므로 추후에 공부를 더 해봐야 할 것 같다.
참고링크 : 
* https://angeloyeo.github.io/2020/08/05/ROC.html
* https://heeya-stupidbutstudying.tistory.com/entry/ML-%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9C-%EC%98%A4%EC%B0%A8%ED%96%89%EB%A0%AC-PRC-ROC-AUC

- 🧐 Decision Theory란?
	- 데이터 기반의 모델에서 모델의 생성과 평가와 관련된 결정을 내리는 이론

- 🧐 Receiver Operating Characteristic Curve란 무엇인가?
	- True-Positive 와 False-Negative 를 클래스 분류를 위해 threshold에 대한 이진분류기의 성능을 한번에 표시한 것. X축은 False-Positive, Y축은 True-Positive이다.
	- 이 그래프가 왼쪽위로 붙을수록 좋은 성능의 모델이다.
![300](https://i.imgur.com/XGcHrvN.png)


- 🧐 Precision Recall에 대해서 설명해보라
	- Precision : $\frac{TP}{TP + FP}$
		- 정밀도라고 하며 특정 모델이 True 라고 분류한 데이터들 중 실제로 True 인 데이터의 비율
	- Recall : $\frac{TP}{TP+FN}$
		- 재현율이라고 하며 특정 모델의 정답(True를 True라고 분류하고 False를 False라고 분류한 것) 중 True 인 정답의 비율을 말한다.
![](https://i.imgur.com/eSkzwly.png)
참고 링크 : https://sumniya.tistory.com/26

- 🧐 Precision Recall Curve란 무엇인가?
	- Precision을 Y축 Recall 을 X축으로 하는 그래프이다. 이 그래프가 오른쪽 위로 더 붙어있을수록 좋은 모델이다.
![300](https://i.imgur.com/EoreJ4A.png)

- 🧐 Type 1 Error 와 Type 2 Error는?
	* Type1 Error : 실제 값이 Negative 인데 Positive 라고 판단하는 에러
		* confusion matrix 에서 FP 인 경우를 의미한다.
	* Type2 Error : 실제 값이 Positive 인데 Negative 라고 판단하는 에러
		* confusion matrix 에서 FN 인 경우를 의미한다.
![](https://i.imgur.com/hgW4nLS.png)


   ---
   
1. [[meaning of KL-Divergence, entropy]]
2. https://velog.io/@hya0906/2022.03.05-ML-Entropy%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80
수학적인 관점(정보론)으로 해석한 것은 1번 링크 참고
ML 관점에서 해석한 것은 2번 링크 참고
아래 내용들은 왠만하면 1번 링크에 정리가 되어 있으므로 참고하라

- 🧐 Entropy란 무엇인가?
	- information 의 기대값이다. ML 관점에서는 클래스간의 불균형을 수치로 나타낸 것이다.
- 🧐 KL-Divergence란 무엇인가?
	- 두 확률분포가 얼마나 차이가 있는지 나타내는 수치이다.
- 🧐 Mutual Information이란 무엇인가?
	- 두 확률분포가 독립이 아닐 때 얼마나 서로 독립적인지를 수치로 나타낸 것이다.
- 🧐 Cross-Entropy란 무엇인가?
	- Cross-Entropy란 두 사건이 얼마나 일치하는지를 나타내는 값이다. 여기서 두 사건이랑 prediction value와 actual value를 의미한다.
- 🧐 Cross-Entropy loss 란 무엇인가?
	- 위의 cross-entropy 를 사용한 loss function 으로 마지막 출력층에 주로 사용하며 이전층까지 연산된 값을 0~1사이의 확률로 표시해 준다.

    ---
    

- 🧐 Generative Model이란 무엇인가?
	- 주어진 학습데이터의 분포를 따르는 유사한데이터를 생성하는 모델

- 🧐 Discriminative Model이란 무엇인가?
	- 

- 🧐 Discrinator function이란 무엇인가?
	- 
    
    ---
참고링크 : 
https://justweon-dev.tistory.com/19
https://bruders.tistory.com/80

- 🧐 Overfitting 이란? [[Answer Post]](https://jrc-park.tistory.com/271)
	- Training 을 위해 투입된 Dataset 에서만 좋은 성능을 보이고 새로운 Dataset 에서는 성능이 좋지 않은 현상
- 🧐 Underfitting이란? [[Answer Post]](https://jrc-park.tistory.com/271)
	- 모델의 복잡도가 지나치게 작아서 학습 자체가 제대로 되지 않은 경우
- 🧐 Overfitting과 Underfitting은 어떤 문제가 있는가?
	- overfitting의 문제점 : 기존의 데이터셋에 대해 완벽에 가까운 성능을 내지만 새로운 데이터는 성능이 매우 떨어진다.
	- underfitting의 문제점 : 기존의 데이터셋에 대해서도 성능이 떨어진다.


- 🧐 Overfitting과 Underfitting의 해결법은?
	- Overfitting : 학습된 데이터의 양을 늘리거나 기존의 데이터셋에 k-fold 같은 cross validation 기법을 적용한다.
	- Underfitting : 학습의 횟수를 늘리거나 좀 더 파라미터가 많은 복잡한 모델을 채택한다.
![400](https://i.imgur.com/4MvVZeh.png)

- 🧐 Regularization이란?
	- overfitting 을 해결하기 위한 기법 중 1개로 지나치게 모델이 복잡할 때(weight 가 지나치게 커서 모델이 복잡) 이 가중치에 제약조건을 줘서 가중치를 줄이는 기법이다.
	 ![](https://i.imgur.com/SkLb97B.png)
	* Lasso : 
		* regression 의 경우 MSE 를 줄이는 방향으로 진행한다.
		* 이 때 penalty 를 $\alpha * L_{1}-norm*$ 의 형태로 주게 된다.
		* $\alpha$ 값을 지나치게 키우게 되면 MSE 가 작아진다 : 이 때가 underfitting이다
		* $\alpha$ 값을 지나치게 작게 되면 MSE 가 커진다 : 이 때가 overfitting이다
		* 위 $\alpha$ 를 하이퍼파라미터로 조정하여 overfitting과 underfitting을 피한다.
	$$
	\begin{align*}\\
	MSE+penalty\\
	&= MSE+\alpha*L_{1}-norm\\
	&= \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_{i})^2}+\alpha\sum_{j=1}^{m}|w_j|\\
	\end{align*}
	$$
	$$
	\begin{align*}\\
	\underset{w, b}{\operatorname{argmax}}\{\frac{1}{n}\sum_{n}^{i=1}(y_i-\hat{y_{i})^2}+\alpha\sum_{j=1}^{m}|w_j|\}
	\end{align*}
	$$



	* Ridge : 
		* Lasso 와 기본적인 개념은 같다.
		* 단 Lasso 와 다르게 Ridge 는 penalty 에 L2-norm 을 사용한다.
	 $$
	\begin{align*}\\
	MSE+penalty\\
	&= MSE+\alpha*L_{1}-norm\\
	&= \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_{i})^2}+\alpha\sum_{j=1}^{m}w_j^2\\
	\end{align*}
	$$
		$$
	\begin{align*}\\
	\underset{w, b}{\operatorname{argmax}}\{\frac{1}{n}\sum_{n}^{i=1}(y_i-\hat{y_{i})^2}+\alpha\sum_{j=1}^{m}w_j^2\}
	\end{align*}
	$$

---

- 🧐 Activation function이란 무엇인가?3가지 Activation function type이 있다.
	- 입력신호에 대해 특정함수를 거쳐 출력신호를 결정하는 함수
	- sigmoid : $1/(1+e^{-x})$, ReLU : $max(0, x)$ 등이 대표적이다
	- (각 Activate Function의 사용이유와 장단점은 Gradient 이후 서술함)

- 🧐 CNN에 대해서 설명해보라
	- Convolution Neural Network 의 줄임말로 computer vision 분야에서 많이 쓰이는 기법이다.
	- kernel과 기존의 데이터를 convolution 하여 feature 맵을 추출하여 이미지의 특징을 알아내는 작업이다.
	- 참고링크 : https://ctkim.tistory.com/entry/%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9DConvolutional-neural-network-CNN%EC%9D%B4%EB%9E%80

- 🧐 RNN에 대해서 설명해보라
	- 

- 🧐 Netwon's method란 무엇인가?
	- $f(x) = 0$ 에서 해당 함수의 해를 구하는 방식 중 1개로 ML 에서는 loss 함수의 최소값을 구하기 위해 사용한다.
	- 기본적으로 newton method 는 아래 형태를 띈다.
	  $$
	x_{i+1} = x_{i}-\frac{f(x_i)}{f'(x_i)}
	$$
		위의 식은 특정 Loss 함수에서 y = 0 인 점을 찾을 때 사용된다. 하지만 우리가 찾아야 하는 것은 극소값이다.
		극소값을 찾기 위해서는 미분을 한 번 더 진행하여 $f'(x)=0$ 인 지점을 찾아야 한다. 따라서 식은 아래와 같다.
		$$
			x_{i+1} = x_{i}-\frac{f'(x_i)}{f''(x_i)}
		$$

	* 하지만 ML의 optimization 관점에서 보게 된다면 우리가 

- 🧐 Gradient Descent란 무엇인가?
	- Loss function의 극소값을 찾는 방법중의 1개로 경사의 절댓값이 낮은 방향으로 이동하여 Loss function 의 극속값 즉 Loss function 의 미분값이 0인 지점을 찾는 알고리즘이다.
	- $x_{i+1}=x_{i}-\alpha\frac{df}{dx}(x_{i})$
		(위의 식에서 $x_{i}$ Weight 라고 보면 되고 $f(x)$ 는 Loss function 이라고 보면 된다)

- 🧐 Stochastic Gradient Descent란 무엇인가?
	* BGD(Batch Gradient Descent) 는 데이터셋 전체에 대하여 Gradient Descent 통해 Weight 를 갱신한다. 하지만 SGD(Stochastic Gradient Descent)는 전체 데이터셋을 임의의 $n$ 개의 데이터셋으로 $n$ 번의 weight 갱신을 발생시킨다.

- 🧐 BGD 와 SGD 의 장단점
	- BGD : 안정적으로 수렴하는 경향이 있으나 메모리적으로 비용이 크다
	- SGD : 비교적 업데이트가 불안정하고 진동하는 경향이 있으나 연산이 빠르다.

- 🧐 Local optimum으로 빠지는데 성능이 좋은 이유는 무엇인가?
	* 대부분의 모델에서 Global Opimum을 찾는 것은 불가능한 경우가 많다 . 이 상황에서 Local Optimum에 빠지더라도 실제 모델을 사용하는데는 크게 지장이 없는 경우가 많기 때문이다.

- 🧐 Internal Covariance Shift 란 무엇인가?
	- 레이어를 통과할수록 Batch단위간의 데이터 분포의 차이가 커지는 현상.
	- 이로 인하여 학습에 시간이 많이 소모되며 학습 자체도 불안정해지게 된다.
	- https://cvml.tistory.com/5

- 🧐 Batch Normalization은 무엇이고 왜 하는가?
	- 레이어의 입력부의 데이터들을 평균이 0, 분산이 1인 표준정규분포로 만들어 준다.
	- 이 때 이러한 Batch Normalization을 진행한 데이터를 $\gamma, \beta$  를 변수로 두고 이 두 변수를 학습하게 된다.
	- $BN(X)=\gamma\left((X-\mu_{batch})/{\sigma_{batch}}\right)+ \beta$ 
	- 참고링크 : https://gaussian37.github.io/dl-concept-batchnorm/

- 🧐 Backpropagation이란 무엇인가?
	- 

- 🧐 Optimizer의 종류와 차이에 대해서 아는가?
	- 

+자료 : 딥러닝에 대한 전반적인 프로세스
https://amber-chaeeunk.tistory.com/19

--- 

    

- 🧐 Ensemble이란?
- 🧐 Stacking Ensemble이란?
- 🧐 Bagging이란?
- 🧐 Bootstrapping이란?
- 🧐 Boosting이란?
- 🧐 Bagging 과 Boosting의 차이는?
- 🧐 AdaBoost / Logit Boost / Gradient Boost
    
    ---
    

- 🧐 Support Vector Machine이란 무엇인가?
- 🧐 Margin을 최대화하면 어떤 장점이 있는가?

---

## ***Computer Vison***



---

## _**Linear Algebra**_ 

- 🧐 Linearly Independent란?
- 🧐 Basis와 Dimension이란 무엇인가?
- 🧐 Null space란 무엇인가?
- 🧐 Symmetric Matrix란?
- 🧐 Possitive-definite란?
- 🧐 Rank 란 무엇인가?
- 🧐 Determinant가 의미하는 바는 무엇인가?
- 🧐 Eigen Vector는 무엇인가?
- 🧐 Eigen Vector는 왜 중요한가?
- 🧐 Eigen Value란?
- 🧐 SVD란 무엇인가?→ 중요한 이유는?
- 🧐 Jacobian Matrix란 무엇인가?