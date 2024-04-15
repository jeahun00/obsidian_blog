REF:
1. https://jang-inspiration.com/latent-diffusion-model
2. https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ldm/
3. 

# Abstract

* diffusion model 은 synthesis 에서 탁월한 성능을 보인다.
* 하지만 diffusion model 은 pixel space 에서 직접적으로 동작하기에 아래 문제가 발생한다.
	* <span style='color:#eb3b5a'>많은 GPU 자원소모 + 긴 시간 소요</span>
	* <span style='color:#eb3b5a'>inference 에 많은 자원 소모</span>
* 이에 이 논문에서는 <mark style='background:#2d98da'>pretrained autoencoder 를 이용한 latent space</mark> 를 적용한다.
* 위의 방법론을 통해 아래의 이점을 취한다.
	* <span style='color:#2d98da'>complexity reduction 과 detail preservation 간의 near optimal point 제시</span>
* 또한 architecture 에 <mark style='background:#2d98da'>cross-attention layer 를 추가</mark>함으로써 아래의 이점을 취한다.
	* diffusion model 을 <span style='color:#2d98da'>general conditioning input(e.g. text, bbox)에 대해 powerful하고 flexible 하게 바꿀 수 있었다</span>.
	* 또한 <span style='color:#2d98da'>high-resolution synthesis 를 convolutional 한 방법론을 사용할 수 있도록</span> 만들었다.

* 이러한 방법론들을 추가한 모델을 <mark style='background:#2d98da'>Latent Diffusion Model(LDM)</mark>이라고 칭하였으며, 아래 task 에서 <span style='color:#2d98da'>SOTA 를 달성</span>하며 동시에 <span style='color:#2d98da'>pixel-based DM 대비 낮은 computational resource 와 높은 성능을 획득</span>하였다.
	1. unconditional image generation
	2. text-to-image synthesis
	3. super-resolution

# 1.  Introduction
* GAN
	* unstable 하며, 변동성이 제한된다(input image 에서 크게 달라지지 않는다).
	* 복잡한, multi-modal distribution 으로의 확장이 어렵다.
* Diffusion
	* super-resolution, class-conditional image synthesis 에서 좋은 성능을 보임
	* <mark style='background:#2d98da'>likelihood-based model 을 선택</mark>함에 따라 <span style='color:#2d98da'>stable 하며</span>, <mark style='background:#2d98da'>parameter sharing 을 통해</mark> AR과는 달리 <span style='color:#2d98da'>수십억개의 parameter 없이도 복잡한 분포를 modeling</span> 하였다.

## Democratizing High-Resolution Image Sythesis
* Diffusion model 은 likelihood-based model 중 하나이다.
* 이 모델은 mode-covering 한 특징 때문에 미세한 detail modeling 에 과도한 용량을 써야 하는 경향이 존재한다.
	* [mode convering 에 대한 reference](https://jaejunyoo.blogspot.com/2017/02/unrolled-generative-adversarial-network-1.html)
* 위와 같은 이유로 DM(Diffusion based Model)은 아래 단점을 가진다.
	1. <span style='color:#eb3b5a'>매우 많은 computing 자원(e.g. v100 으로 150-1000일 정도 소요)이 소모</span>된다.
	2. 위의 이유로 인해 <span style='color:#eb3b5a'>엄청난 양의 탄소발자국을 남긴다</span>.
	3. 또한 <span style='color:#eb3b5a'>sampling 에도 많은 자원이 소모</span>된다(A100 GPU 에서 50k sample 을 생성하는데 약 5일 소요)
	4. 또한 <span style='color:#eb3b5a'>일반 연구자들이 접근하기 힘들 정도의 computing 자원이 필요</span>하여, 소수의 연구자들만이 연구할 수 있다.
* 이 논문에서는 성능을 저하시키지 않는 선에서 접근성을 강화하는 것을 논의한다. 

## Departure to Latent Space
* Likelihood-based generative model 은 아래 2개의 과정을 거친다.
	1. **perceptual compression stage**
		* high frequency detail 은 삭제되지만 little semantic information 은 유지
	2. **semantic compression stage**
		* 실제 generative model 이 학습을 하는 단계
		* 이 과정에서 generative model 은 data 의 semantic+conceptual 을 학습한다.
* 위의 과정을 통해 ==perceptually equivalent== 하며 ==compute 자원을 효율적으로 사용==하는 **space 를 찾는 것이 목적**이다.

* 이 논문에서도 위의 과정을 따르지만 일부 다르다.
	1. **train an autoencoder**
		* 이전의 모델들은 pixel space 에서 학습했지만,
		* 이 논문에서는 <mark style='background:#2d98da'>latent space 에서 학습</mark>함
		* 이 과정을 통해 <span style='color:#2d98da'>과도한 spatial compress 에 의존하지 않아도 된다</span>.
		* 이 과정은 <span style='color:#2d98da'>더 나은 spatial compression 을 제공</span>한다.
	2. **single network pass**
		* 위의 과정을 통해 줄어든 complexity 로 <span style='color:#2d98da'>single network pass 만으로 image generation 이 가능</span>해졌다.
		* 이러한 모델을 <mark style='background:#2d98da'>Latent Diffusion Model(LDM)</mark>라고 부르기로 했다.

* 위의 과정은 아래의 큰 장점을 가져다 준다.
1. <span style='color:#2d98da'>universal autoencoding 단계를 한번만 훈련</span>하면 됨
2. 위의 이유로 <span style='color:#2d98da'>여러 DM에 재사용하거나 아예 다른 task에 적용가능</span>.
3. 이를 통해 <span style='color:#2d98da'>image-to-image, text-to-image task 에 확장 가능</span>
	* text-to-image 의 경우 U-net backbone 에 transformer 를 연결하는 기능 제공
![](Pasted%20image%2020240325172012.png)

### 정리
* 이 모델의 장점을 정리하자면 아래와 같다.
	1. faithful 하고 detail 한 이미지를 생성하며 megapixel image 에 효율적으로 사용가능
	2. 적은 computational cost 로 경쟁력있는 성능을 보임. 또한 inference cost 도 크게 감소시킴
	3. reconstruction 과 generative ability 간의 세밀한 weigh 조정이 필요 없다.
	4. cross-attention 을 적용함으로써 multi-modal 학습을 가능하게 한다.

# 2. Related Work

## Generative Models for Image Synthesis
* GAN
	* <span style='color:#2d98da'>high resolution image 를 좋은 perceptual quality 로 생성가능</span>
	* <span style='color:#eb3b5a'>하지만 최적화 되기는 어려우며 전체 데이터 distribution 을 포착하는데 어려움을 겪음</span>


* Likelihood-based method
	* 양질의 density estimation 을 진행하여 최적화를 더 잘 수행함


* VAE(Variational AutoEncoder) / Flow-based model
	* <span style='color:#2d98da'>high resolution image 에 대한 효율적인 synthesis 를 가능하게 한다</span>
	* <span style='color:#eb3b5a'>sample 의 성능은 GAN base model 에 비해 떨어진다</span>

* AutoRegressive Model(ARM)
	* <span style='color:#2d98da'>density estimation 에서 좋은 성능을 보임</span>
	* <span style='color:#eb3b5a'>computational demanding 하다</span>
	* <span style='color:#eb3b5a'>sequential 한 sampling 과정이 low resolution 으로 제한시킨다</span>
      
## Diffusion Probabilistic Model
* <span style='color:#2d98da'>density estimation 과 sample quality 에서 좋은 성능</span>을 보임
* 하지만 <mark style='background:#eb3b5a'>DM 을 pixel 공간에서 evaluating and optimizing 을 진행</mark>하는 것은 
	* <span style='color:#eb3b5a'>low inference speed</span> 와 
	* <span style='color:#eb3b5a'>high training cost </span>를 야기한다.
* 위의 2가지 문제점을 LDM 으로 해결한다.
	* LDM 은 품질을 저하시키지 않으며 효율적인 inference time 을 보인다.

# 3. Method
* 고해상도 이미지 합성에 대한 diffusion model 학습의 계산 요구량을 낮추기 위해 
* diffusion model이 해당 손실 항을 적게 샘플링하여 perceptual하게 관련 없는 detail들을 무시할 수 있지만 
* 여전히 <span style='color:#eb3b5a'>픽셀 space에서 계산 비용이 많이 드므로 계산 시간과 컴퓨팅 리소스가 많이 필요</span>하다.
* 이에 이 논문에서는 <mark style='background:#2d98da'>generative learning phase 에서 compressive 단계를 완전히 분리 해 낼 것을 제안</mark>한다.
> 참고:
> 기존의 모델들(e.g. DDPM, DDIM, guided Diffusion 등)은 pixel level 에서 forward, backward process 를 진행한다.
* 위의 방식은 아래의 장점들을 제공한다.
	1. image space 에서 직접 image 를 처리하는 것이 아니라 compression 과정을 통해 압축된 latent space 에서 처리되기에 훨씬 더 효율적인 DM 을 얻는다.
	2. Unet 구조를 차용하여 spatial 한 정보를 수월하게 처리한다.
	3. Latent space 는 여러 생성모델을 훈련하는 데 쉽게 확장가능하고 downstream task 에도 활용가능하다.

## 3.1. Perceptual Image Compression
* Perceptual Compression Model
	* perceptual loss 와 patch-based adversarial objective 의 조합으로 학습되는 autoencoder 로 구성됨
	* 위의 방식으로 구현함으로써 local realism 이 적용되어 reconstruction 시 image manifold 에 국한된다.
	* 또한 아래의 문제를 방지할 수 있다.
		* pixel space loss(L1, L2 loss) 에만 의존하여 blur 가 발생하는 현상

### Encoding 과정
$$
\begin{align*}
encoder:z=\mathcal{E}(x)\\
x\in\mathbb{R}^{H\times{W}\times{3}}
\end{align*}
$$
* encoder $\mathcal{E}$ 가 $x$ 를 latent representation $z=\mathcal{E}(x)$ 로 encoding 한다.
	* $x$ : RGB 값을 가지는 이미지
	* $\mathcal{E}$ : input image 로 부터 새로운 feature를 생성하는 encoder
* encoder 는 input image 를 factor $f$ 만큼 downsampling 한다.
$$
\begin{align*}
factor:f=H/h=W/w\\
\end{align*}
$$
* 위의 $f$ 는 아래의 값을 가질 수 있으며 이는 추후 Experiment에서 보였다.
$$
f=2^m(m\in\mathbb{N})
$$


### Decoding 과정 
$$
\begin{align*}\\
\tilde{x}=\mathcal{D}(z)=\mathcal{D}(\mathcal{E}(x))\\
z\in\mathbb{R}^{h\times w\times c}
\end{align*}
$$
* decoder $\mathcal{D}$ 가 latent representation $z$ 로 부터 새로 만들어질 이미지 $\tilde{x}$ 를 재구성한다.
	* $\tilde{x}$ : RGB 값을 가지는 **생성된** 이미지
	* $\mathcal{D}$ : input latent representation 으로 부터 새로운 image 를 생성하는 decoder

### Regularizations
* latent space 의 분산이 커지는 것을 방지하기 위해 두 종류의 regularization 을 실험했다.
	1. KL-regulariztion
		* VAE 와 유사하게 학습된 standard normal distribution 에 대해 약간의 패널티를 부여한다.
	2. VQ-regularization
		* decoder 내에서 vector quantization layer 를 사용한다.
		* 위는 VQGAN 과 유사한 형태이다.

* 위와 같은 regularizaion 기법들은 latent space $z=\mathcal{E(x)}$ 의 2차원 구조와 잘 병합될 수 있다.
* 이전의 autoencoder 를 사용하는 기법들은 latent sapce 를 1D ordering 에 의존하여 $z$ 안의 구조를 무시했다.
* 하지만 우리의 모델은 $x$ 의 detail 을 더 잘 포착할 수 있다.


### Objective Function of Autoencoder
* 이 논문에서 autoencoder 는 [이 논문](https://openaccess.thecvf.com/content/CVPR2021/papers/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.pdf)의 GAN 방법론을 따른다.
$$
\begin{equation}L_{\textrm{Autoencoder}} = \min_{\mathcal{E}, \mathcal{D}} \max_{\psi} \bigg( L_{\textrm{rec}} (x, \mathcal{D}(\mathcal{E}(x))) - L_{\textrm{adv}} (\mathcal{D}(\mathcal{E}(x))) + \log D_\psi (x) + L_{\textrm{reg}} (x; \mathcal{E}, \mathcal{D}) \bigg)\end{equation}
$$
* **GAN Term**
	* 아래 2개의 term 은 GAN 과 관련된 term 들이다.
	* $L_{adv}(\mathcal{D}(\mathcal{E}(x)))$ : 
		* GAN 의 objective function 의 adversarial term
		* [이 링크](https://process-mining.tistory.com/169) 참고
	* $\log{D_{\psi}(x)}$ : 
		* patch-based discriminator $D_{\psi}$ 가 원본 이미지 $x$ 와 재구성된 이미지 $\mathcal{D}(\mathcal{E}(x))$ 를 판별하는 Term
* **VAE Term**
	* 아래 2개의 term 은 VAE 와 관련된 term 들이다.
	* $L_{rec}(x,\mathcal{D}(\mathcal{E}(x)))$ : Reconstruction Error
		* VAE 의 reconstruction error 와 동일
		* [이 링크](https://velog.io/@gunny1254/Variational-Auto-Encoder-VAE) 참고
	* $L_{reg}(x;\mathcal{E},\mathcal{D})=D_{KL}(q_{\mathcal{E}}{(z|x)}||{\mathcal{N}(z;0,1)})$ : Regularization Term
		* $q_{\mathcal{E}}{(z|x)}=\mathcal{N}(z;\mathcal{E_\mu},\mathcal{E}_{\sigma^2})$ 
		* $\mathcal{N(z;0,1)}$
		* 위의 2개의 term 의 KL divergence 로 regularization 효과를 부여
		* 단, 위의 KL term 은 매우 작은 수치로 적용 : 약 $10^{-6}$
		* VAE 의 Regularization(KL divergence) Term 과 유사하다.
	
> 위의 수식에 관한 것은 정확하지가 않다.
> 따라서 논문의 appendix 를 참고하고 다른 조사를 좀 더 해야 할 것 같다.
## 3.2. Latent Diffusion Models

### Diffusion Models
$$
\begin{equation}L_{DM} = \mathbb{E}_{x, \epsilon \sim \mathcal{N} (0,1), t} \bigg[ \\| \epsilon - \epsilon_\theta (x_t, t) \\|_2^2 \bigg]\end{equation}
$$
* $x_{t}$ : input $x$ 에서 noise 가 추가된 부분
* uniform 하게 $\{1,...,T\}$ 로 sampling 함
* 위 수식은 score-based diffusion 의 term 과 유사하다.
* ddpm 이나 score-based diffusion 이나 수식이 유사하기에 이렇게 표기함

### Generative Modeling of Latent Representation
* $\mathcal{E,D}$로 구성된 학습된 perceptual compression model 을 high-frequency, 감지할 수 없는 detail 들을 생략할 수 있는 latent space 에 접근 가능
* 이러한 형태의 학습은 아래의 이유로 인해 likelihood-base model 에 적합
	* 데이터에 더 중요한 semantic 정보에 더 집중할 수 있음
	* computational efficiency 가 증가
$$
\begin{equation}
L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t} \bigg[ \| \epsilon - \epsilon_\theta (z_t, t) \|_2^2 \bigg]
\end{equation}
$$
* 이전 연구들이 굉장히 압축된 discrete latent space에서 autoregressive attention 기반 transformer 모델에 의존했지만, LDM은 이미지별 inductive bias를 활용할 수 있다. 
* 여기에는 2D convolutional layer들로 UNet을 구성하는 것이 포함되며, 다음과 같은 reweighted bound를 사용하여 perceptual하게 가장 관련성이 높은 비트에 목적 함수를 더 집중시키는 것도 포함된다.

## 3.3. Conditioning Mechanism
* diffusion model 은 $p(z|y)$ 형태의 conditional distribution 을 모델링 할 수 있다.
* 위를 통해 conditional denoising distribution $\epsilon_\theta(z_t,t,y)$ 를 구현함으로써 아래를 제어한다.
	* $y$ 는 input data 이다.
	* $y$ 는 text, semantic map, image 등이 될 수 있다.
		* 즉 $y$ 가 text 면 전체 모델이 text-to-image 가 되는 형식

* 이전 연구들에서는 diffusion model 의 <span style='color:#eb3b5a'>class label, blurred 외의 다른 condition(e.g. text, semantic map, image, etc ...)과의 결합이 연구되지는 않았다.</span>
* 이 논문에서는 <mark style='background:#2d98da'>cross attention 을 통해</mark> <span style='color:#2d98da'>다양한 유형의 condition 을 추가하는 방법론</span>을 제시한다.
---
### 참고: 
* [ref link](https://vds.sogang.ac.kr/wp-content/uploads/2023/01/2022%ED%95%98%EA%B3%84%EC%84%B8%EB%AF%B8%EB%82%98_%EC%9C%A0%ED%98%84%EC%9A%B0.pdf)
* Cross Attention 이란?
	* self attention 과 mechanism 은 동일하나 input 의 출처가 다른 경우
	* 이 논문에서 self attention : 
		* 아래 이미지의 왼쪽에서 Q, K, V 는 latent space 에서 추출된 feature
	* 이 논문에서 cross attention : 
		* 아래 이미지의 오른쪽에서 Q 는 latent space feature
		* K, V 는 condition(e.g. text, image, semantic map, etc...) 으로 부터 추출된 feature
![300](Pasted%20image%2020240327151248.png)
---
$$
\begin{align*}
\textrm{Attention}(Q, K, V) = \textrm{softmax}(\frac{QK^T}{\sqrt{d}}) \cdot V \\
\end{align*}
$$
$$
\begin{align*}
Q = W_Q^{(i)} \cdot \varphi_i (z_t), \quad K = W_K^{(i)} \cdot \tau_\theta (y), \quad V = W_V^{(i)} \cdot \tau_\theta (y)
\end{align*}
$$
* $W_Q^{(i)}$ : 
	* leanable projection matrix
	* $W_Q^{(i)} \in \mathbb{R}^{d \times d_r}$ 
* $\varphi_i(z_t)$ :
	* $\epsilon_\theta$ 를 구현하는 UNet 의 intermediate representation
	* $\varphi_i(z_t)\in{\mathbb{R}^{N\times d^i_\epsilon}}$

* $W_K^{(i)}$ : 
	* leanable projection matrix
	* $W_K^{(i)} \in \mathbb{R}^{d \times d_r}$ 
* $\tau_\theta(y)$ :
	* $y$ 를 다양한 종류의 modality 로부터 전처리 하기 위해 $y$ 를 encoder $\tau_\theta$ 를 이용하여 intermediate representation $\tau_\theta(y)$ 로 projection 시킨다.
	* $\tau_\theta(y)\in{\mathbb{R}^{M\times d_r}}$

* $W_V^{(i)}$ : 
	* leanable projection matrix
	* $W_V^{(i)} \in \mathbb{R}^{d \times d_\epsilon^r}$  
* $\tau_\theta(y)$ :
	* $y$ 를 다양한 종류의 modality 로부터 전처리 하기 위해 $y$ 를 encoder $\tau_\theta$ 를 이용하여 intermediate representation $\tau_\theta(y)$ 로 projection 시킨다.
	* $\tau_\theta(y)\in{\mathbb

$$
\begin{equation}L_{LDM} := \mathbb{E}_{\mathcal{E} (x), y, \epsilon \sim \mathcal{N} (0, 1), t} \bigg[|| \epsilon - \epsilon_\theta (z_t, t, \tau_\theta (y))||_2^2 \bigg]\end{equation}
$$
* 위에서 사용한 cross-attention 을 적용하여 conditional LDM 수식을 완성한다.
* 이 수식에서 $\epsilon_\theta$ 와 $\tau_\theta$ 는 jointly optimize 된다.

To preprocess $y$ from various types of modalities, 
$y$ is projected into an intermediate representation $\tau_\theta(y)$ using the encoder $\tau_\theta$


The intermediate representation 
of UNet implementing $\epsilon_\theta$


$IS(G) = \exp(\mathbb{E}_{x \sim p_a} D_{KL}(p(y|x) || p(y)))$


Ultimately, the goal is to create $\epsilon(z_{t},y,t)$ from $\epsilon(z_t,t)$ with the added condition $y$