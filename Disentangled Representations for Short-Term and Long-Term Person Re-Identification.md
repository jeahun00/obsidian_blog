![[img_store/Pasted image 20240209215747.png]]

### 3.2.1. Identity-Related Loss 

$$
\mathcal{L}_R = - \sum_{c=1}^{C} \sum_{k=1}^{K} q_c^k \log p(c|w_c^{k} \phi_R^k (I^k))
$$
* $w_c^k$ : classifier parameter
* $q_c^k$ : 
	* if label $c$ is identity of the image $I^k$(image 의 identity label 과 $c$ 가 일치 할 때, 즉 $c=y$ 일 때) : 1
	* otherwise : 0
* $p(c|w_c^{k^\top} \phi_R^k (I^k)) = \frac{\exp(w_c^{k} \phi_R^k (I^k))}{\sum_{i=1}^{C} \exp(w_i^{k} \phi_R^k (I^k))}$ : softmax function

### 3.3.1. Identity-Shuffling Loss

$$
\mathcal{L}_s = \sum_{i,j \in \{a,p\}} \left\| I_i - G\left(\phi_R(I_j) \oplus \phi_U(I_i)\right) \right\|_1
$$
* if $i = j$ : enforcing combination of identity-related, unrelated
* if $i \neq j$ : $E_r$ extract similar identity related features.


### 3.3.2. Part-Level Shuffling Loss

$$
\mathcal{L}_{PS} = \sum_{\substack{i,j \in \{a,p\} \\ i \neq j}} \left\| I_i - G\left(S\left(\phi_R(I_i), \phi_R(I_j)\right) \oplus \phi_U(I_i)\right) \right\|_1
$$

* $G$ : generator
* Why $i\neq j$ ? : i = j 일 경우 같은 feature 이므로 shuffle 의 의미가 없다

### 3.3.3. Identity-Unrelated Loss
$$
\mathcal{L}_U = \sum_{k=1}^{K} D_{KL}\left( \phi_U^{k}(I^{k}) \middle\| \mathcal{N}(0,1) \right)
$$
* $\phi_U^{k}(I^{k})$ : Identity-Unrelated feature
* $\mathcal{L}_U$ 를 최소화하려면 위의 $\phi_U^{k}(I^{k})$ 를 정규분포에 가깝도록 학습시켜야 한다.
* 정규분포로 학습을 시키는 이유는 unrelated feature 의 영향을 줄이기 위함이다.
$$
\mathcal{L}_U = \sum_{k=1}^{K} \frac{(\phi_R^k(I^k) - \mu_R^k)(\phi_U^k(I^k) - \mu_U^k)}{\sigma_R^k \sigma_U^k}
$$
* 또한 Pearson correlation coefficient 를 최소화하여 identity-related, unrelated 간의 특징이 서로 무관해지도록 학습을 진행. 
* 이러한 수치는mutual information 을 이용해 측정 가능
* Mutual information 은 두 분포가 완전히 Independent 하면 0, 완전히 dependent 하면 1 이다.

### 3.3.4. Domain and Class Losses

![[img_store/Pasted image 20240211035126.png]]