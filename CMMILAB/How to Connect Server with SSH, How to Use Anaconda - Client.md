# 1. SSH connect

### 1.1. window
* window 같은 경우 터미널 구성과 그 명령어들이 독자적인 것들이 많다.
* 따라서 일반적인 bash shell 기반으로 사용할 수 있게 해 주는 mobaXterm 설치를 권장한다.
* install link : https://mobaxterm.mobatek.net/
![AltText|500](https://i.imgur.com/QKli2Lo.png)
ssh 접근 방법은 아래 링크를 참고
https://iter.kr/mobaxterm-ssh-%EC%84%A4%EC%A0%95-%EB%B0%8F-%EC%A0%91%EC%86%8D/

![AltText|500](https://i.imgur.com/GXF7N4p.png)

* MobaXterm 을 켠 후 상단의 `+` 버튼을 누르면 새 창이 나온다.
* 여기에 아래 명령어를 치면 접근이 가능하다.
```bash
$ ssh [user name]@[server com ip]
# example
$ ssh ubuntu@10.11.41.235
```
### 1.2. mac
* mac os 같은 경우 bash shell 혹은 Linux shell 과 대부분의 명령어를 공유한다.
* 따라서 mac 은 기본터미널을 써도 무방하다.
* 하지만 zsh 를 이용한 iterm2 를 쓰는 것을 추천한다.
* iterm2 설치 링크 : https://iterm2.com/
* homebrew 설치 링크 : https://brew.sh/

ssh 접근 명령어 : 
```bash
$ ssh [user name]@[server com ip]
# example
$ ssh ubuntu@10.11.41.235
```
![AltText|500](https://i.imgur.com/ZmzX8lL.png)

### 1.3. connect 가 에러가 뜨는 경우
1. 오랜 기간 컴퓨터가 꺼져 있을 때 ip 가 바뀔수가 있다.
리눅스 컴퓨터를 직접 켜서 터미널을 키고 ifconfig 라고 치면 해당 ip 가 나온다.
```bash
$ ifconfig
```

![AltText|500](https://i.imgur.com/P5jLEoV.png)
2. 인터넷 연결문제
이 경우에는 리눅스 컴퓨터에 직접 접근하여 인터넷이 켜져 있는지 확인해야 함

# 2. Anaconda 사용법
현재 서버컴퓨터에 default 로 `tf1.14` 가 실행되어 있다.
anaconda 가 실행되고 있는지 확인하는 방법은 터미널 시작 부분 제일 앞에 괄호로 무엇이 있는지 확인하면 된다.
![AltText|500](https://i.imgur.com/t2oQQI0.png)
conda 가 깔려 있는 상태에서 base 환경이 있을 때는 아래 화면이 나온다.
![AltText|500](https://i.imgur.com/FVMTSuL.png)

### 2.1. conda 환경 생성
```bash
$ conda create -n [가상환경이름] python=[python 버전]
``` 
* [가상환경이름] : 본인이 생성하고자 하는 가상환경 이름
* [python 버전] : python 버전 지정
cf)기본적으로 conda 에는 python3 가 깔린다.
이 때 python 명령어를 python3 로 연결해 주는 역할이다.
즉 원래 python test.py 를 입력하면 python2 로 실행이 되지만 위의 옵션을 넣어주면 python, pip 를 쓰더라도 내가 지정한 python 버전이 실행된다.
* 사용예
```bash
$ conda create -n condaTest python=3.9
```
* 실행화면
![AltText|500](https://i.imgur.com/MzkgKsJ.png)

### 2.2. conda 환경 실행
```bash
$ conda actiavte [가상환경이름]
# 예시
$ conda activate condaTest
```
* 실행화면
![AltText|500](https://i.imgur.com/DDYeml0.png)


### 2.3. conda 환경 비활성화
```bash
$ conda deactivate
```
* 실행화면
![AltText|500](https://i.imgur.com/TxE9LPy.png)

### 2.4. conda 환경에서 특정 package 설치
아래 예시에서 matplotlib 는 pip 로, numpy 는 conda 로 깔아볼 것이다.
1. conda install
```bash
$ conda install [package name]
# ex
$ conda install numpy
```
package name 에 해당하는 라이브러리 다운로드
![](https://i.imgur.com/OD7Hk6u.png)

2. pip3 install (or pip install)
```bash
$ pip3 install [package name]
# ex)
$ pip3 install matplotlib
```
package name 에 해당하는 라이브러리 다운로드
![](https://i.imgur.com/VihODLw.png)

* 위의 pip, conda 는 다운받는 위치가 다르다
* 하지만 대부분의 라이브러리는 2곳 모두에서 다운이 가능하다.
* 하지만 conda 의 경우 conda 가상환경에 특화되어 있어서 dependency 문제가 덜하다.
* 하지만 두 개의 명령어 모두 왠만해서는 문제가 발생하지 않는다.

### 2.5. 설치된 라이브러리 확인
* pip, conda 둘 다 확인이 가능하다.
1. pip list
```bash
$ pip list
```
![](https://i.imgur.com/aTZrRHt.png)
2. conda list
```bash
$ conda list
```
![](https://i.imgur.com/qh3jnxI.png)

* <mark style='background:#eb3b5a'>주의!</mark>
	* pip list 로 조회하는 경우 conda 로 설치된 패키지가 뜨지 않을 수 있다.
	* conda list 로 조회하는 경우 Channel 에 conda로 설치했는지 pip로 설치했는지 표시가 되기 때문에 conda list 로 조회하는 것을 추천한다.

### 2.6. requirements.txt 로 일괄설치
pip 의 경우 논문마다 설치해야 하는 패키지가 많기도 많고 다른 경우가 있다.
이럴 경우 모든 라이브러리를 pip 로 일일히 설치하는 것은 너무 소모적이다.
따라서 requirements.txt 를 제공하는 코드는 이 파일로 일괄설치가 가능하다.
* requirements.txt
```
# Usage: pip install -r requirements.txt

# Base ----------------------------------------
matplotlib>=3.2.2
numpy>=1.18.5,<1.24.0
opencv-python>=4.1.1
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0,!=1.12.0
torchvision>=0.8.1,!=0.13.0
tqdm>=4.41.0
protobuf<4.21.3

# Logging -------------------------------------
tensorboard>=2.4.1
# wandb

# Plotting ------------------------------------
pandas>=1.1.4
seaborn>=0.11.0

# Export --------------------------------------
# coremltools>=4.1  # CoreML export
# onnx>=1.9.0  # ONNX export
# onnx-simplifier>=0.3.6  # ONNX simplifier
# scikit-learn==0.19.2  # CoreML quantization
# tensorflow>=2.4.1  # TFLite export
# tensorflowjs>=3.9.0  # TF.js export
# openvino-dev  # OpenVINO export

# Extras --------------------------------------
ipython  # interactive notebook
psutil  # system utilization
thop  # FLOPs computation
# albumentations>=1.0.3
# pycocotools>=2.0  # COCO mAP
# roboflow
```

* 위의 텍스트 파일 안에 있는 패키들을 일일이 까는 것은 불편하기에 아래 명령어를 치면 저 안에 있는 패키지들의 종속성을 맞춰 깔아준다
```bash
$ pip3 install -r requirements.txt
```
![](https://i.imgur.com/RmNlq6n.png)

가상환경삭제, 가상환경 조회 등은 아래 블로그 참고
https://jeahun10717.tistory.com/23