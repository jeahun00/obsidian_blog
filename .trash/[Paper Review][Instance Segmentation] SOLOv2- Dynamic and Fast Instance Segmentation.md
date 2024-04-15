# Abstract

* 이 논문에서는 instance segmentation 기법인 SOLOv2 를 제시한다.
* SOLOv2 는 SOLO 논문의 기본원칙들을 따른다.
* SOLOv2 에서 제안하는 핵심기법은 아래 2가지 이다.
	1. Mask kernel prediction, mask feature learning 을 분리
	2. 기존의 NMS 대신 matrix NMS 사용

> 참고:
> 이 논문은 기존의 [SOLO](https://arxiv.org/pdf/1912.04488.pdf])에서 개선한 논문이므로 완벽한 이해를 위해서는 SOLO(SOLOv1)을 읽고 오는 것을 추천한다.


# 1. Introduction

* Object detection 과는 달리 Instance segmentation 은 pixel level 의 정확도가 필요하다.
* 또한 이전의 Instance segmentation 기법들은 object detection task 이후의 작업으로 여겨져 순수한 instance segmentation 기법은 비교적 덜 연구되었다.
* 하지만 최근에 제안된 SOLO(Segmenting Objects by LOCation)은 아래 2가지 의의를 가진다.
	1. 순수하게 Instance segmentation작업만을 진행한다.
	2. FCN을 이용하여 2개의 하위 task(kernel prediction, feature learning)로 분리하여 학습하며 이는 anchor-free, fully-convolutional, grouping-free 한 성질을 가진다.

# 2. Proposed Method: SOLOv2

### Architecture of SOLOv1
