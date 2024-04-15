# 본격적으로 들어가기 앞서...

Computer Vision 에서 주요하게 다루는 4가지 문제가 있다.
1. Classification
2. Semantic Segmentation
3. Object Detection
4. Instance Segementation

CS231n 강의에서 Object Detection 은 비교적 자세하게 다루는 반면 Instance Segmentation 은 그 내용이 빈약했다. 이에 Instance Segmentation 의 주요 논문인 Mask R-CNN 을 정리하고자 한다.

### 참고!
---

Object Detection 에 해당하는 내용은 추후에 정리를 할 예정이다.
위의 분야의 기초는 아래 강의에서 다졌다.

Michigan Univ EECS 498-007 / 598-005 Lecture Link
https://www.youtube.com/watch?v=9AyMR4IhSWQ&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=16

위의 링크는 CS231n 의 강사와 같은 사람이 미시간 대학에서 진행한 강의이다. CS231n 은 2017 년 강의여서 최신 트렌드에 대한 내용이 빈약한 것에 반해 위의 강의는 최신 업데이트가 더 되어 있다.

---

# introduction

* 먼저 Mask R-CNN 의 기반이 되는 Faster R-CNN 에 대해 간략히 짚고 넘어가고자 한다.
* Faster R-CNN 은 기존의 Fast R-CNN 에서 RoI 를 통한 영역 후보 선정을 RPN 이라고 하는 Network 기반구조를 추가한 구조이다.

* Faster R-CNN 의 구조는 아래와 같다.
![[Pasted image 20231225133715.png]]

* Faster R-CNN 은 Fast R-CNN 에서 RoI 로 추천영역을 추출하던 것을 RPN 이라는 네트워크 기반으로 추출하는 것으로 바뀐 구조이다.
* Faster R-CNN 에서 Mask R-CNN 구조로 가며 바뀐 점은 아래와 같다.
	1.  Faster R-CNN 의 Classification, Localization(Bounding Bod Regression) branch 에 mask branch 가 추가됨
	2. RPN 전에 FPN 이 추가됨
	3. Image Segmentation 의 masking 을 위해 RoI Align 이 RoI Pooling 을 대채

# Mask R-CNN 

![[Pasted image 20231227103718.png]]

1. 800~1024 사이즈로 이미지를 resize해준다. (using bilinear interpolation)**
2. Backbone network의 인풋으로 들어가기 위해 1024 x 1024의 인풋사이즈로 맞춰준다. (using padding)
3. ResNet-101을 통해 각 layer(stage)에서 feature map (C1, C2, C3, C4, C5)를 생성한다.
4. FPN을 통해 이전에 생성된 feature map에서 P2, P3, P4, P5, P6 feature map을 생성한다.
5. 최종 생성된 feature map에 각각 RPN을 적용하여 classification, bbox regression output값을 도출한다.
6. output으로 얻은 bbox regression값을 원래 이미지로 projection시켜서 anchor box를 생성한다.
7. Non-max-suppression을 통해 생성된 anchor box 중 score가 가장 높은 anchor box를 제외하고 모두 삭제한다.
8. 각각 크기가 서로다른 anchor box들을 RoI align을 통해 size를 맞춰준다.
9. Fast R-CNN에서의 classification, bbox regression branch와 더불어 mask branch에 anchor box값을 통과시킨다.

REF:
* https://ganghee-lee.tistory.com/40
위의 링크에서는 **Bilinear Interpolation**, **RoI Align** 을 설명하고 있다.
이 부분은 중요한 부분이라서 추후에 따로 정리를 할 예정이다.