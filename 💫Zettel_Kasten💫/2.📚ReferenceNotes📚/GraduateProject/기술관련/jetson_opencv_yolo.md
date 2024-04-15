---
tistoryBlogName: jeahun10717
tistoryTitle: "[IOT/DL] Jetson Nano 에 YoLO, opencv with cuda 설치"
tistoryVisibility: "3"
tistoryCategory: "1175570"
tistorySkipModal: true
tistoryPostId: "76"
tistoryPostUrl: https://jeahun10717.tistory.com/76
---
> 이 글은 jetson nano 4gb 모델에서 진행했으며 jetpack 4.6.1 image 설치 후 진행음을 알린다.

## 1. opencv 4.5.1 with cuda 설치

### 1.1. package update 및 설치

opencv 설치 전 관련 패키지를 먼저 설치해야 한다.
```bash
sudo apt update
sudo apt install -y python3-pip python-dev python3-dev python-numpy python3-numpy 
sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
sudo apt install -y qt5-default
sudo apt install -y build-essential cmake git unzip pkg-config libswscale-dev
sudo apt install -y libcanberra-gtk* libgtk2.0-dev
sudo apt install -y libtbb2 libtbb-dev libavresample-dev libvorbis-dev libxine2-dev 
sudo apt install -y curl
```

### **1.2. Math, Video, image format package download**

```bash
sudo apt install -y libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libmp3lame-dev libtheora-dev libfaac-dev libopencore-amrnb-dev
sudo apt install -y libopencore-amrwb-dev libopenblas-dev libatlas-base-dev
sudo apt install -y libblas-dev liblapack-dev libeigen3-dev libgflags-dev 
sudo apt install -y protobuf-compiler libprotobuf-dev libgoogle-glog-dev  
sudo apt install -y libavcodec-dev libavformat-dev gfortran libhdf5-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libv4l-dev v4l-utils qv4l2 v4l2ucp libdc1394-22-dev
```

### **1.3. Download OpenCV & Contribs Modules**

opencv 4.5.1. 을 sh 파일을 받아서 설치하는 방법이 있긴 하지만 자잘한 에러가 많이 발생해서 아래 처럼 zip 파일을 받아서 하는 것이 더 낫다는 판단을 했다.

```bash
curl -L https://github.com/opencv/opencv/archive/4.5.1.zip -o opencv-4.5.1.zip
curl -L https://github.com/opencv/opencv_contrib/archive/4.5.1.zip -o opencv_contrib-4.5.1.zip
```

### **1.4. Unzip packages**

```bash
unzip opencv-4.5.1.zip 
unzip opencv_contrib-4.5.1.zip 
cd opencv-4.5.1/
```

### **1.5. Create the build packages**

```bash
mkdir build
cd build
```

### **1.6. Build Opencv using Cmake**

cuda 관련 옵션들을 포함한 cmake : 3~10m 소요

```bash
cmake         -D WITH_CUDA=ON \
        -D ENABLE_PRECOMPILED_HEADERS=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.1/modules \
        -D WITH_GSTREAMER=ON \
        -D WITH_LIBV4L=ON \
        -D BUILD_opencv_python2=ON \
        -D BUILD_opencv_python3=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.1/modules \
        -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
        -D CUDA_ARCH_BIN="5.3" \
        -D CUDA_ARCH_PTX="" \
        -D WITH_CUDNN=ON \
        -D WITH_CUBLAS=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D ENABLE_NEON=ON \
        -D WITH_QT=OFF \
        -D WITH_OPENMP=ON \
        -D WITH_OPENGL=ON \
        -D BUILD_TIFF=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_TBB=ON \
        -D BUILD_TBB=ON \
        -D WITH_EIGEN=ON \
        -D WITH_V4L=ON \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D INSTALL_C_EXAMPLES=ON \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D BUILD_opencv_python3=TRUE \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_EXAMPLES=OFF ..
```

### **1.7. Compile the OpenCV with Contribs Modules**

opencv 를 dependency 외 필요한 모듈, 라이브러리를 포함하여 컴파일 하는 부분이다.

이 과정 자체가 1~2h 정도 소요되기에 위의 1.1. ~ 1.6. 의 과정을 한 번 체크 하고 make -j4 를 하는 것을 추천한다.

```bash
# core 수 출력
nproc

make -j[core 수]
ex) make -j4

sudo make install
sudo ldconfig
```

### **1.8. Check the OpenCV Version on Terminal**

```bash
python3 -c 'import cv2; print(cv2.__version__)'
```

위의 결과가 아래와 같으면 정상적으로 설치된 것이다.

```
4.5.1
```

opencv 에서 cuda 가속이 설치되어 있는지 확인하는 것은 아래 yolo 설치가 끝난후 일괄적으로 서술하겠다.

---

## **2. torch with cuda and YoLOv7**

이 파트에서는 YoLOv7 의 구동을 위한 torch(with cuda)와 Jetson Nano 에서 구동을 원할하게 하게 위한 스왑메모리 설정등을 설명할 것이다.

### **2.1. yolo project setup**

`./yolo` 폴더를 만들고 그 안에 yolov7 깃헙을 clone 한다.

```bash
mkdir yolo 
cd yolo 
git clone https://github.com/WongKinYiu/yolov7
```

### **2.2. pip3 virtual environment setup**

global 에 설치를 진행해도 되지만 여러 에러를 만났기에 빠르게 환경설정을 다시 할 수 있는 가상환경을 택했다.

jetconda 라고 하는 jetson 용 anaconda 가상환경이 존재하지만 자잘한 에러가 너무 많아서 virtualenv 를 설치했다.

```bash
sudo apt update
sudo apt install python3-pip
sudo pip3 install virtualenv virtualenvwrapper
```

### 2.3. virtualenv environment setup

```bash
# 터미널에서 텍스트를 편집하기 위한 에디터 설치
sudo apt update 
sudo apt install nano 
nano ~/.bashrc
```

위의 `nano ~/.bashrc` 를 실행하면 bash shell 실행 시 초기실행되는 명령어들이 저장되어 있다.
여기서 아래 내용을 붙여넣고 `[ctrl] + [x] -> [y] -> [enter]` 를 하여 저장하고 나오면 된다.

```bash
# virtualenv and virtualenvwrapperexport WORKON_HOME=$HOME/.virtualenvsexport VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3source /usr/local/bin/virtualenvwrapper.sh
```

yolov7 가상환경 생성
```bash
# yolov7 is the name of our virtual environment. You can use any other name.
mkvirtualenv yolov7 -p python3 
```

yolov7 가상환경 활성화
```bash
# yolov7 is the name of our virtual environment
workon yolov7 
```

workon 환경이 설치가 제대로 되었다면 아래와 같은 화면이 나온다. 
```bash
jetson1@jetson1-desktop:~$ workon yolov7
(yolov7) jetson1@jetson1-desktop:~$
```

yolov7 가상환경 비활성화
```bash
deactivate
```

### **2.4. 가상환경에 opencv 라이브러리 링크 걸기**

이글의 1번에 따라 설치된 opencv 는 system-wide 에 설치가 되어 있다.
따라서 system-wide 로 깔려 있는 opencv 라이브러리들을 우리가 설정한 가상환경에 연결할 필요가 있다. 
```bash
# workon 공유라이이브러리 설정 폴더
cd ~/.virtualenvs/yolov7/lib/python3.6/site-packages/
# opencv 설치 경로
ln -s /usr/local/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-aarch64-linux-gnu.so cv2.so
```
opencv 설치경로에 따라 위의 2번째 라인은 달라질 수 있다. 아마 대부분 아래 2중 1개일 듯 하다.
```bash
/usr/local/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-aarch64-linux-gnu.so cv2.so

/usr/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-aarch64-linux-gnu.so cv2.so

```

이제 opencv 연결이 끝났으므로 처음에 생성한 yolo 로 돌아가면 된다.

### **2.5. yolo 동작을 위한 dependency lib 설치(torch 포함)**

yolo 동작을 위한 pytorch 를 설치해야 한다. 그 전에 여러 관련한 여러 라이브러리들을 먼저 설치한다.
```bash
sudo apt install libfreetype6-dev
sudo apt-get install python3-dev
```

* numpy, matplotlib 설치
```bash
pip3 install --upgrade pip setuptools wheel
pip3 install numpy==1.19.4
pip3 install matplotlib
```
위는 matplotlib, numpy 를 우선 설치하여 추후 requirements로 자동설치 될 때 dependency 문제를 해결하기 위함이다.

* requirements.txt 설정
```bash
nano requirements.txt
```
requirements.txt 에서 이미 설치된 라이브러리들, dependency 문제로 따로 설치를 진행할 라이브러리들은 주석처리하여 자동설치를 방지한다.
-> matplotlib, numpy, opencv-python, torch, torchvision, thop 주석처리
```bash
# matplotlib>=3.2.2
# numpy>=1.18.5
# opencv-python>=4.1.1
.
.
.
# torch>=1.7.0,!=1.12.0 
# torchvision>=0.8.1,!=0.13.0
.
.
.
# thop  
# FLOPs computation # --> LOCATED ON THE BOTTOM OF requirements.txt
```

* requirments 설치
```bash
pip3 install -r requirements.txt
```

* dependency lib 설치
```bash
sudo apt-get install libopenblas-base libopenmpi-dev
```

* torch 설치
```
pip3 install -U future psutil dataclasses typing-extensions pyyaml tqdm seaborn 

pip3 install Cython  

wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl  pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
pip3 install thop
```

* torch 설치 확인
```bash
python -c "import torch; print(torch.__version__)"
```
결과창
```
1.8.0
```

* torchvision 설치
```
sudo apt install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
pip3 install --upgrade pillow
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.9.0
python3 setup.py bdist_wheel # Build a wheel. This will take a couple of mincd dist/pip3 install torchvision-0.9.0-cp36-cp36m-linux_aarch64.whl
cd ..
cd ..
sudo rm -r torchvision # Now remove the branch you cloned
```

* torchvision 설치확인
```bash
python -c "import torchvision; print(torchvision.__version__)"
```
결과창
```
0.9.0
```

### 2.6 yolov7 실행

* weight 파일 다운로드
```bash
# make weights directory
mkdir weights
cd weights

# Download tiny weightswget 
https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt  
# Download regular weightswget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```
내가 참조한 블로그에서는 `./yolo/yolov7` 메인에 바로 weight 파일을 설치했다. 
이렇게 했을 때 weight를 찾지 못하는 것 같아서 그냥 `weights` 폴더를 만들어서 설치했다.

* yolov7 실행
```bash
python3 detect.py --weights [weight 파일 위치] --conf 0.25 --img-size 640 --source [사진,동영상,웹캠등]

# ex)
python3 detect.py --weights ./weigths/yolov7-tiny.pt --conf 0.25 --img-size 640 --source 0

```

아마 jetson nano 에서 위의 명령어를 치면 돌아가지 않을 확률이 높다.
아무리 tiny weight 라 하더라도 yolo 자체가 워낙 큰 모델이라서 안돌아갈 것이다.
만약 돌아간다면 그냥 그대로 쓰면 되고 아니면 swap memory를 설정을 해줘야 한다. 
이는 아래 설명을 해 두었다.

### **2.7. swap memory 설정**

```bash
# 이미 했다면 안해도 됨
sudo apt-get update
sudo apt-get upgrade

sudo apt-get install nano
sudo apt-get install dphys-swapfile
 
# /sbin/dphys-swapfile 파일 open
sudo nano /sbin/dphys-swapfile

## Swap파일의 값이 다음과 같도록 값을 추가하거나, 파일 내 주석을 해제
# CONF_SWAPSIZE=4096
# CONF_SWAPFACTOR=2
# CONF_MAXSWAP=4096
 
# 값을 수정 후 [Ctrl] + [X], [y], [Enter]를 눌러 저장하고 닫으면 됨.
 
# /etc/dphys-swapfile 파일 open
sudo nano /etc/dphys-swapfile

## Swap파일의 값이 다음과 같도록 값을 추가하거나, 파일 내 주석을 해제
# CONF_SWAPSIZE=4096
# CONF_SWAPFACTOR=2
# CONF_MAXSWAP=4096
 
# 값을 수정 후 [Ctrl] + [X], [y], [Enter]를 눌러 저장하고 닫으면 됨.
 
# 재부팅
sudo reboot
```

이 이후 아래 명령어로 yolo를 돌리면 돌아갈 것이다.
```bash
python3 detect.py --weights ./weigths/yolov7-tiny.pt --conf 0.25 --img-size 640 --source 0
```

### **2.8. jtop 설치**
yolo를 설치했을 때 이제 gpu, cpu 활용률이 궁금할 수 있다.
top 명령어 에서는 gpu 활용률이 나오지도 않고 cpu, ram 활용률이 이상하게 찍히는 경우가 많았다.
따라서 jetson 전용 top 프로그램인 jtop 을 설치한다.

```bash
# 이미 했다면 생략ㅇㅇ
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python-pip

# jetson-stats 설치
sudo -H pip install -U jetson-stats

# 재부팅 or restart 경고 뜨면 그거 실행하면 됨
sudo reboot

# jetpack 버전 확인 및 cpu, 메모리 cuda, opencv 정보까지 확인 가능
jtop
```

![](https://i.imgur.com/KSJSrvD.png)
### **2.9. opencv, ros, torch 설치 확인 및 cuda 가속설정 확인 소스코드**

opencv, 
opencv cuda 가속, 
torch, torchvision, torch/torchvision cuda 가속,
ros, rospy,
위의 라이브러리들이 동작하는지 일일히 확인하는게 귀찮아서 아래 python 파일을 만들었다.
```python
import cv2
import rospy
import torch
import torchvision 

print("=========================")
print("cv2 버전 : ")
print(cv2.__version__)
print("=========================")
print("cv2 waitKey() 동작 확인 : ")
print(cv2.waitKey(0))
print("=========================")
print("opencv2 cuda 가속확인 : ")
print(cv2.cuda.getCudaEnabledDeviceCount())
print("=========================")
print("rospy 동작확인 : ")
print("이 파일이 실행되면 import 된거임ㅋ")
print("=========================")
print("torch 동작 확인 / gpu 사용 확인 : ")
print("torch version : ", torch.__version__)
print("CUDA is available!")
print("torch version : ", torch.__version__)
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0))
print("torchvision version : ", torchvision.__version__)
print("=========================")
```


