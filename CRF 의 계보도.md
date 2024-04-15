![](Pasted%20image%2020240315105406.png)
![](Pasted%20image%2020240315112934.png)
* 첫번째 사진에서 볼 수 있듯이, CRF 모델의 등장은 네 단계를 거쳤습니다. 
* 첫 번째로, 베이즈 정리와 조건부 독립 가정을 기반으로, <mark style='background:#eb3b5a'>나이브 베이지안 모델(NB)</mark> (Maron 1960)이 제안되었고, 입력 특징을 고려할 때 <mark style='background:#eb3b5a'>Logistic Regression(LR)</mark> (Verhulst 1838; Pearl and Reed 1920)로 발전했습니다. 
* 이를 바탕으로, 다중 특징 조건 제약이 도입되어 <mark style='background:#eb3b5a'>Maximum Entropy Model(MEM)</mark> (Jaynes 1957a, b; Berger 1997)이 구축되었습니다. 
* 그 후, 관측의 독립 가정과 균질 마르코프 가정에 따라 시간 순서가 나이브 베이지안 모델에 고려되어 <mark style='background:#eb3b5a'>Hidden Markov Model(HMM)</mark> (Rabiner and Juang 1986; Rabiner 1989)이 생성되었습니다. 
* 그런 다음 Hidden Markov Model과 Maximum Entropy Model의 장점이 결합되어 <mark style='background:#eb3b5a'>MEMM (Mccallum et al. 2000)</mark>이 구축되었습니다. 마지막으로, 입력 feature의 global normalization 의 고려를 바탕으로 <mark style='background:#2d98da'>CRF 모델이 제안</mark>되었습니다 (Lafferty et al. 2001).