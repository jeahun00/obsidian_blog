
* REF:
	* https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/segment-anything/
	* https://int-erest.tistory.com/entry/General-Segment-AnythingSAM-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0#%F0%9F%9B%A0%EF%B8%8F%20SAM%EC%9D%80%20%EC%A7%84%EC%A7%9C%20foundation%20model%20%EC%9D%B8%EA%B0%80%3F%20-%20Zero-shot%20Transfer%20%08Ability-1
	* https://da2so.tistory.com/78
	* https://fakecan.tistory.com/96

---
# Abstract

* 이 논문은 **segmentation anything** 이라는 목적을 가진 논문이다.
* 이 논문은 2가지 작업을 진행했다.
	1. **1-billion** 에 달하는 **segment dataset 을 구축**하였다.
	2. **promptable** 하며 **transfer zero-shot learning** 을 이용하여 아래 성과를 달성하였다.
		* <span style='color:#2d98da'>zero-shot 임에도 경쟁력 있는 성능을 보였다</span>.
		* 몇몇 성능에서 <span style='color:#2d98da'>fully supervised 보다 더 좋은 성능</span>을 보였다.

![](Pasted%20image%2020240329171347.png)
# 1. Introduction

 * NLP 에서 zero-shot, few-shot model 의 성능은 impressive 하다.
 * 이 model 들은 web 에서 모은 방대한 dataset 으로 학습하였을 때 매우 좋은 성능을 보인다.
	 * 심지어 fine-tunded model 과도 비슷한 성능을 지닌다.
 * 또한 이러한 모델들은 매우 generalized 하다.

* 하지만 vision task 에서는 이러한 작업에 대한 연구가 비교적 덜 이루어 졌다.
* 이에 이 논문에서는 <mark style='background:#2d98da'>image segmentation task</mark> 에 대해<span style='color:#2d98da'>proptable</span> 하며 <span style='color:#2d98da'>강하게 generalized</span> 될 수 있는 방법론을 제시한다.
* 이 방법론에 대한 구성은 아래 3개로 이루어지며 아래 3개의 구성은 서로 영향을 끼친다.
	1. **Task** : zero-shot generalization 에 맞는 task 를 찾는다.
		* promptable segmentation task
	2. **Model** : 위의 task 에 맞는 model 을 제시한다.
		* 실시간으로 segmentation mask 를 출력할 수 있는 model
	3. **Data** : 위의 Model 과 Task 에 적합한 Data 를 찾는다.
		* diverse, large-scale data
* 또한 segmentation 을 위한 충분한 web-scale 의 데이터가 부족한 문제가 있다.
* 이러한 문제를 해결하기 위해 이 논문에서는 **data-engine** 을 제시한다.

## Task
* 이 논문의 task 는 **promptable segmentation task** 이다.
* 이는 사용자가 입력하는 여러 종류의 **prompt 를 받아** 그에 적합한 **segmentation mask 를 출력**하는 것이 목표인 task 이다.
	* prompt 의 종류: image, text, spatial information, etc...
	* 프롬프트가 일부 모호하더라도 segmentation mask 는 단일하게 출력이 되어야 한다.
![500](Pasted%20image%2020240330160726.png)

## Model
* promptable segmentation task 를 위한 model 은 아래 3가지 조건을 만족해야 한다.
	1. flexible prompts 에 대응 할 수 있어야 한다.
	2. mask 를 거의 real-time 으로 생성할 수 있어야 한다.
	3. ambiguity-aware 해야 한다.(모호한 상황도 처리가 가능해야 한다.)

* 위의 3가지 조건을 만족하는 model 을 SAM(Segment Anything Model) 이라 하며 아래 과정을 따른다.
	1. 강력한 **image encoder** 가 **image 를 embedding**
	2. **prompt encoder** 가 **prompt 를 embedding**
	3. 위의 **두 embedding information** 을 segment mask 를 예측하는 **lightweight mask decoder 에서 병합**한다.
* SAM 을 <mark style='background:#2d98da'>image encoder 와 fast prompt encoder / mask decoder 로 분리</mark>함으로써 <span style='color:#2d98da'>동일한 image embedding 을 다른 prompt 에서도 재사용</span>할 수 있다.
* SAM 은 point, box, mask 에 해당하는 prompt 에 초점을 맞추었다.
* 또한 text-prompt 에 대한 결과도 제공한다.
* 또한 SAM 의 ambiguity 를 처리하기 위해 1개의 prompt 에 대해 여러개의 mask 를 prediction 하도록 설계하였다.
![500](Pasted%20image%2020240330183649.png)

## Data engine
* SAM 의 학습을 위해 많은 데이터셋을 확보할 필요성이 있었다.
* 하지만 online 에서 segment mask 를 포함하는 데이터의 양이 턱없이 부족했다.
* 이에 이 논문에서는 **Data engine** 을 제시한다.
* model 과 함께 model-in-the-loop dataset annotation 을 구축하였다.(아래 그림 참고)
![500](Pasted%20image%2020240330183635.png)
* 이러한 Data engine 은 아래 3가지 단계를 거쳐 annotating 을 진행한다.
	1. assisted-manual : 
		* **[interactive segmentation](https://medium.com/aimmosubscribe/%EC%9D%B4%EA%B2%83%EB%8F%84-%EC%84%B8%EA%B7%B8%EB%A9%98%ED%85%8C%EC%9D%B4%EC%85%98-%EC%A0%80%EA%B2%83%EB%8F%84-%EC%84%B8%EA%B7%B8%EB%A9%98%ED%85%8C%EC%9D%B4%EC%85%98-ac55832885ae)** 과 유사한 tool 을 제공받아 SAM 으로 segmentation 을 진행
	2. semi-automatic : 
		* 객체가 있을 것으로 예측되는 부분에서 **자동**으로 mask 를 생성하고 
			* 이 부분은 위의 assisted-manual 에서 SAM 이 만들어낸 결과이다.
		* 나머지 객체에 대해서는 annotator 가 segmentation 을 수동으로 작업한다.
	3. fully-automatic : 
		* SAM 이 32x32 image 에 대해 grid point 를 부여하고 각 point 에 대해 유효한 mask 를 fully automaic 하게 예측한다.

# 2. Segment Anything Task
 * 이 논문은 NLP 의 next token prediction task 에서 영감을 받았다.
 * 따라서 NLP 와 유사한 기법을 많이 사용한다.

## Task
* possible prompt : 
	* foreground/background points, rough box, rough mask, free-from text, etc ...
* promptable task:
	* 위의 possible prompt 를 segmentation mask 로 치환하는 task 라고 요약할 수 있다.
	* 또한 NLP 와 유사하게 모호한 prompt 에서도 합리적인 단일한 결과를 출력하도록 설계할 것이다.
		* (아래 그림 참고) : 두번째 줄의 사람을 예시로 들어보자
		* 초록색 point 에 대해 segment 를 진행해야 할 때 사람으로 분류할지 가방으로 분류할지가 모호하다. 하지만 SAM 은 이러한 모호성을 해소해줄 수 있다.
![500](Pasted%20image%2020240330201348.png)

## Pre-training
* Promptable segmentation task는 각 학습 샘플에 대한 일련의 프롬프트(ex. 점, 박스, 마스크)를 시뮬레이션하고 모델의 마스크 예측을 ground-truth와 비교하는 자연스러운 사전 학습 알고리즘을 제안한다. 
* 충분한 사용자 입력 후에 결국 유효한 마스크를 예측하는 것이 목표인 interactive segmentation과 달리 이 방법을 적용한다. 
* 프롬프트가 모호한 경우에도 모든 프롬프트에 대해 항상 유효한 마스크를 예측하는 것이 목표이다. 
* 이를 통해 데이터 엔진에서 요구하는 자동 주석을 포함하여 모호성이 포함된 use case에서 사전 학습된 모델이 효과적임을 보장한다. 
* 저자들은 이 task를 잘 수행하는 것이 어렵고 전문적인 모델링 및 학습 loss 선택이 필요하다는 점에 주목하였다.

## Zero-shot transfer

* 직관적으로 사전 학습 task는 모델이 inference 시간에 모든 프롬프트에 적절하게 응답할 수 있는 능력을 부여하므로 하위 task는 적절한 프롬프트를 엔지니어링하여 해결할 수 있다. 
	* 즉, 다른 task 로의 확장이 용이하다.
* 예를 들어 고양이에 대한 boundary box detector가 있는 경우 detector의 box 출력을 모델에 프롬프트로 제공하여 고양이 instance segmentation를 해결할 수 있다. 
* 일반적으로 다양한 실용적인 segmentation task는 프롬프트로 캐스팅될 수 있다.

## Discussion

* Prompting과 composition은 단일 모델을 확장 가능한 방식으로 사용하여 잠재적으로 모델 설계 시 알려지지 않은 task를 수행할 수 있도록 하는 강력한 도구이다. 
* 이 접근 방식은 다른 foundation model이 사용되는 방식과 유사하다. 
* 저자들은 프롬프트 엔지니어링과 같은 기술로 구동되는 composition 가능한 시스템 설계가 고정된 일련의 task를 위해 특별히 학습된 시스템보다 더 다양한 애플리케이션을 가능하게 할 것으로 예상하였다. 
* composition의 관점에서 promptable segmentation과 interactive segmentation을 비교하는 것도 흥미롭다. 
* interactive segmentation model은 인간 사용자를 염두에 두고 설계되었지만 promptable segmentation을 위해 학습된 모델은 더 큰 알고리즘 시스템으로 구성될 수 있다.

* 요약 : 즉, SAM 은 prompt base 로 학습을 하기에 다른 task 로의 전이가 용이하고 다른 task 와의 병합도 고려할 수 있다.

# Segment Anything Model 
![](Pasted%20image%2020240330202216.png)
* 위의 그림과 같이 SAM 은 3가지 요소로 이루어져 있다.
	1. image encoder
	2. flexible mask decoder
	3. fast mask decoder

## Image encoder
* MAE pre-trained ViT 를 사용한다.
	* MAE 란?
	* CVPR 2022 에 accept 된 meta 의 논문 중 하나로 [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf) 논문에서 제시한 방법론이다.
	* 이 논문에서는 아래 과정을 따른다.
		1. input image 를 patch 단위로 grid 로 나눈다.
		2. 나누어진 patch 들 중 일부를 masking 하고 다시 복구하는 transformer 를 학습한다. 
		3. 이렇게 학습된 transformer 를 이용한 모델은 feature extractor 로서 좋은 성능을 보인다.
	* 아래는 MAE 논문에 있는 도식이다.
![400](Pasted%20image%2020240331154722.png)
![700](Pasted%20image%2020240331154807.png)
* image encoder 는 이미지당 한 번 실행되며 model 을 prompt 하기 전에 적용할 수 있다.

## Prompt encoder
* prompt 를 2가지로 구분하여 설계한다.
	1. sparse data : point, box
		* 각 prompt 유형에 대해 learned embedding 으로 합산된 positional encoding 을 사용하여 표현 
	2. sparse data : text
		* CLIP 모델의 text encoder 로 text 를 표현.
	3. dense : mask
		*  마스크는 CNN을 사용하여 임베딩되고 image embedding(from image encoder)과 함께 element-wise하게 합산된다.

## Mask Decoder
* 마스크 디코더는 image embedding, prompt embedding, ouput token 을 마스크에 효율적으로 매핑한다. 
* Transformer 디코더 블록을 수정하고 dynamic mask prediction head를 사용한다. 
* 수정된 디코더 블록은 모든 임베딩을 업데이트하기 위해 Prompt Self-Attention과 Cross-Attention을 두 방향(Prompt-to-Image Embedding, Image-to-Prompt Embedding)으로 사용한다. 
	* 아래 그림에서 token-to-image attention, image-to-token attention
* 두 블록을 실행한 후 이미지 임베딩을 업샘플링하고 MLP는 출력 토큰을 dynamic linear classifier로 매핑한 다음 각 이미지 위치에서 마스크 전경 확률을 계산한다.
![](Pasted%20image%2020240331165948.png)
  
* Detail of lightweight mask decoder. 
	* 두 개의 레이어 디코더가 cross-attention을 통해 image embedding과 prompt token 모두를 업데이트한다. 
	* 그런 다음 image embedding이 업스케일되며, 여기서 업데이트된 ouput token을 사용하여 동적으로 마스크를 예측한다. 
	* (그림의 명확성을 위해 미표시: 모든 attention layer에서, postiional encoding 이 이미지 임베딩에 추가되며, 전체 원본 프롬프트 토큰(위치 인코딩 포함)이 토큰 쿼리와 키에 다시 추가됩니다.)

## Resolving ambiguity 
* ambiguity 를 해소하기 위해 여러개의 mask 를 예측하여 평균을 내는 방식을 사용한다.
* mask 를 다른 depth 3 개를 이용하는 것(3개의 mask)이 효과적임을 논문에서 제시한다.
* 이 3개의 depth는 whole, part, subpart 로 나뉜다.(아래 그림 참고)
![500](Pasted%20image%2020240331171222.png)

## Efficiency
* 모든 SAM 의 과정은 1번의 image 에 대해 50ms 내의 시간을 보장한다.
* 이는 real time 을 가능하도록 한다.

## Loss and training
* focal loss 와 dice loss 의 linear combination

# 4. Segment Anything Data Engine
* 위에서부터 지속적으로 언급했듯, image segmentation 을 위한 dataset 은 그 수가 너무 부족하다.
* 이에 이 논문에서는 3가지 단계를 거쳐 11억개의 mask dataset 을 수집한다.
	1. Assisted-manual stage
	2. Semi-automatic stage
	3. Fully automatic stage
## Assisted-manual stage
* SAM 이 지원하는 브라우저를 통해 전문적인 annotator 가 라벨링을 진행했다.
* model 이 먼저 간략하게 segment 를 진행한 이미지를 통해 labeling 을 진행한다.
* 이 stage 에서는 중요도가 높은 객체를 우선적으로 labeling 한다.

## Semi-automatic stage
* mask 의 다양성을 위해 눈에 잘 띄지 않는 물체를 labeling 하는 것을 목표로 하는 stage 이다.
* 

## Fully automatic stage