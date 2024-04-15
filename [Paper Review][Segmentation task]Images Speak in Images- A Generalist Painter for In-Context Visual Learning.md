REF:
* [논문리뷰 링크](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/painter/)

* [In Context Learning(ICL) 이란](https://velog.io/@hundredeuk2/In-context-Leaning)
# Abstract
* In-context learning 을 NLP 에 도입함으로써 다양한 task 를 손쉽게 처리할 수 있었다.
* 하지만 <span style='color:#eb3b5a'>vision task 에서는 이러한 In-context learning 을 도입하고자 하였으나 여러 문제점</span>이 있었다.
	* vision task 의 <span style='color:#eb3b5a'>output representation 이 너무 다양</span>하기에 <span style='color:#eb3b5a'>외부 domain task 로의 transfer 가 어려운 문제</span>가 있었다.

* 이 논문에서는 위의 문제점을 해결하기 위해 generalist 한 모델인 <mark style='background:#2d98da'>Painter</mark> 를 제시한다.
* <mark style='background:#2d98da'>Painter</mark> 는 아래 장점을 가진다.
	* high, low level image 처리 모두 가능
	* 다양한 vision task 에서 경쟁력 있는 성능 달성
	* 몇몇 난해한 vision task 에서 일반화된 모델보다 더 큰 성능 달성

# Introduction

* NLP 에 ICL 의 도입은 NLP 분야에서 generalist model 을 만들 수 있는 발판이 되었다.
* 그렇기에 **vision task 에 ICL 을 도입하고자** 하였으나 **NLP 와 2가지의 차이점이 존재**했기에 <span style='color:#eb3b5a'>도입이 어려웠다</span>.
	1. 서로 다른 task 에서의 output space 의 통일성
		* **NLP** 의 여러 task 의 output space 는 그 형식이 discrete language token 의 sequence 로 동일하게 유지될 수 있다.
		* **CV** 의 여러 task 의 output space 는 그 형식을 통일시키기가 어렵다.
			* e.g. 
				* semantic segmentation task 에서는 HxWx1 의 semantic map 형태
				* object detection 에서는 coord of object, class of object 의 pair 형태
	2. input/output space 의 통일성
		* **NLP** 에서는 input space 와 output space 가 같다.
		* **CV** 에서는 input space 와 output space 가 다르다.
			* e.g.
				* object detection 에서는 input 이 HxWx3 의 이미지
				* output 은 coord, class 의 pair 형태
				* 즉 input, output 의 형식이 다르다.
* 위와 같은 이유로 CV 에 ICL 을 도입하기 위해 CV 에 NLP 기법을 사용하는 방식을 도입하기도 하였다.
* <mark style='background:#2d98da'>하지만 이 논문에서는 image 를 설명할 수 있는 것은 image 밖에 없다는 사실을 강조한다.</mark>
* 또한 여러 **dense prediction vision task** 를 **image inpainting 으로 간주**하고자 한다.
* 위의 image inpainting 을 적용하기 위해 input 과 output 을 다음과 같이 정의한다.
	* input image : raw rgb image(HxWx3)
	* output image : 3-channel tensor(HxWx3)
		* 각 task 에 맞게 output 은 일부 조정을 거친다.
		* task : `depth estimation`, `human keypoint detection`, `semantic segmentation`, `instance segmentation`, `image denoising`, `image deraining`, `image enhancement`
* 위의 i**nput/output image** 를 말 그대로 **이어붙여(stitching)** **하나의 큰 이미지로 취급**한다.
* inference 할 때의 출력 이미지 역시 input/output image pair 를 가지며 output image 는 비어 있는 상태이다.
* 위의 비어 있는 <mark style='background:#2d98da'>output image 를 image inpaining 으로 채워 넣는 것</mark>이 이 논문에서 제시하는 방법이다.
![](Pasted%20image%2020240403223006.png)
* 위와 같은 방식으로 vision task 에 ICL 기법을 적용시켰다.
* 이 논문에서는 input/output image(image/image label image pair) 그 자체로 어떤 task 인지 인지한다.

# 3. Approach
* 이 논문에서 제시한 frame work
	* 다양한 vision task(특히 dense vision task)를 image inpatining 으로 재구성 하는 것이 핵심이다.
	* 즉 input, output space 모두를 image space 로 취급한다.

## 3.1. Redefining Output Spaces as "Images"
* $\mathrm{x}$  : input image
	* size : $H\times W\times3$ 
* $\mathrm{y}^t$ : image $\mathrm{x}$ 와 그 task $t$ 에 대한 task Ground Truth
	* size : Task 에 따라 다름
* $\hat{\mathrm{y}}^t$ : task Ground Truth $\mathrm{y}^t$ 를 input space 와 같은 space 로 맞춘 task output
	* task ground truth 이 나타내고자 하는 representation 과 동일
	* 하지만 그 size 가 input image space 와 동일
	* size : $H\times W\times3$ 

* input image 의 특정 한 pixel 이 $\mathrm{x}_{i,j}$ 라고 하면, 그 GT label 은 $\hat{\mathrm{y}}^t_{i,j}$ 이다.

* 정리하자면 원래 **input image $\mathrm{x}$** 에 대한 **GT label $\mathrm{y}^t$** 는 서로의 **space 가 일치하지 않는다**.
	* (task $t$ 마다 output space 가 다르기 때문)
* 하지만 **image inpainting 으로의 통합**을 위해 **GT label $\mathrm{y}^t$** 를 **image space 로 변환한 $\hat{\mathrm{y}}^t$** 를 **GT label 로 취급**하고 사용한다.

* 이 논문에서는 총 7개의 task 를 다루게 된다.

### Monocular depth estimation
* Dataset : NYUv2
* $\mathrm{y}^t_{i,j}$ : per-pixel GT depth 
	* 값의 범위 : \[0, 10\]
	* 각 값의 의미 : meter
* $\hat{\mathrm{y}}^y_{i,j,0},\hat{\mathrm{y}}^y_{i,j,1},\hat{\mathrm{y}}^y_{i,j,2}$ : per-pixel GT "RGB" image
	* 값의 범위 : \[0, 255\]
	* 각 값의 의미 : bit(0~8bit)

* 변환과정
	1. mapping : $\hat{\mathrm{y}}^t_{i,j,0}=\lfloor\mathrm{y}^t_{i,j}\times\frac{255}{10}\rfloor$ 
	2. output 의 3개의 channel $\hat{\mathrm{y}}^t_{i,j,0},\hat{\mathrm{y}}^t_{i,j,1},\hat{\mathrm{y}}^t_{i,j,2}$ 에 위에서 사용한 동일한 값을 대입한다.
	3. inference 시에는 3개의 채널에 평균을 내고 inverse linear transform 을 수행하여 \[0,10\] 사이의 값을 가지는 depth map 으로 치환한다.

### Semantic segmentation
* RGB space 가 $\mathrm{L}$ 개의 semantic label 을 나타내는 것이 목표

* Dataset : ADE-20K
* $\mathrm{y}^t_{i,j}$ : semantic map
	* 값의 범위 : \[0, L\)
	* 각 값의 의미 : semantic label 의 갯수
* $\hat{\mathrm{y}}^t_{i,j,0},\hat{\mathrm{y}}^t_{i,j,1},\hat{\mathrm{y}}^t_{i,j,2}$ : per-pixel GT "RGB" image
	* 값의 범위 : \[0, 255\]
	* 단위 : bit(0~8bit)

* 변환과정
	* $b$ 진수를 이용하여 3개의 자릿수를 만들고 그 3개의 자릿수 안에 L 개의 semantic label 을 담을 수 있어야 한다.
	* mapping : $b=\lceil L^\frac{1}{3}\rceil,m=\lfloor \frac{256}{b} \rfloor$ 
	* $\hat{\mathrm{y}}^t_{i,j,0},\hat{\mathrm{y}}^t_{i,j,1},\hat{\mathrm{y}}^t_{i,j,2}$ 는 각각 1의자리, 10의자리 100의자리 수를 저장한다.
	*  $\hat{\mathrm{y}}^t_{i,j,0}=\lceil{\frac{l}{b^2}}\rceil\times{m}$
	*  $\hat{\mathrm{y}}^t_{i,j,1}=\lceil{\frac{l}{b}}\rceil\mod{b}\times{m}$
	*  $\hat{\mathrm{y}}^t_{i,j,2}=l\mod{b}\times{m}$  
* 즉, 각 semantic label 을 표현할 수 있는 bit masking 을 만든 과정과 유사하다.
* 단, 2진수가 아닌 $b$진수를 이용하는 것이 차이점이다.

* 이 논문에서 사용한 ADE-20K 의 예시
	* ADE-20K 는 150개의 라벨을 가짐
	* 따라서 $b=6$으로 설정
		* 만약 $b=5$ 이면 $444_{(5)}=124_{(10)}$ 이므로 150개보다 적기 때문에 전체를 포함하지 못함
		* 하지만 $b=6$ 이면 $555_{(6)}=215_{(10)}$ 이므로 150개를 전부 포함 가능

### Keypoint detection
* 이 task 는 object 의 위치와 그 object 의 keypoint 의 detail 을 찾는 task 이다.
* 이에 이 논문에서는 heatmap-based top-down pipeline 을 따른다.

* $\mathrm{y}^t$ : heatmap 구조로, object 의 position 정보와 각 keypoint 의 좌표로 구성됨(아래 그림 참고)
![500](Pasted%20image%2020240404164003.png)
* $\hat{\mathrm{y}}^t_{0},\hat{\mathrm{y}}^t_{1},\hat{\mathrm{y}}^t_{2}$ : per-pixel GT "RGB" image
	* $\hat{\mathrm{y}}^t_1,\hat{\mathrm{y}}^t_2$ : 
		* 17개의 keypoint 에 대해 각 keypoint 는 9x9 pixel 로 표현
		* 각 square pixel 은 semantic segmentation 처럼 색상 부여
	* $\hat{\mathrm{y}}^t_0$ : 
		* localization 을 위한 map 으로 object 의 위치를 표현한다.
		* 각 keypoint 의 위치를 중심으로 17x17 pixel 로 표현한다.
		* 각 keypoint 를 중심으로 하는 gaussian distribution 을 적용한다.
			* 값의 범위는 \[0,255\]

### Panoptic segmentation
* Panoptic segmentation = instance segmentation + semantic segmentation
* redefinition 을 좀 더 수월하게 진행하기 위해 instance segementation 과 semantic segmentation 을 각각 진행한 후 합치는 방법을 사용한다
* instance segmentation 에서 각 instance 마다 다른 색을 부여함으로써 RGB 이미지를 만들어 낸다. 
* 하지만 이 경우 어떠한 instance 에 어떠한 color 를 부여할지가 문제가 된다.
	* 논문에서는 random 한 color 부여가 optimize 를 방해한다고 한다.
* 이 때 SOLO 를 이용한다.
![](Pasted%20image%2020240404170457.png)
* SOLO 를 이용하여 이미지를 16x20x20 block 으로 분할하고 각 채널에 할당한다.
* 각 블록에 고정된 색상을 할당하고 instance 의 중심이 그 block 에 해당할 때 그 block 의 색상을 해당 instance 에 부여한다.
* inference 시에는 각 block 을 하나의 커널로 취급하여 이미지 내 각 pixel 과의 거리를 계산한 다음,threshold 를 설정하여 최종 mask 를 획득한다.

### Image restoration
* 이 파트에서는 image denoising, image deraining, low-light image enhancement 3개의 image restoration task 에 대해 설명한다.
* image restoration 같은 경우 input/output 이 원래 RGB space 이므로 따로 작업할 필요가 없다.