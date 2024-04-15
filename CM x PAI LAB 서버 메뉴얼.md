# 1. Ubuntu 설치

참고 : 아래는 mac os 를 기준으로 하였다.
1. 부팅 USB 준비
2. Balena etcher 다운로드(아래 링크 참고)
([https://etcher.balena.io/](https://etcher.balena.io/))
3. Ubuntu iso 이미지 다운로드(연구실은 Ubuntu 20.04 로 통일)(아래 링크 참고)
([https://ubuntu.com/download/desktop](https://ubuntu.com/download/desktop))
4. Balena etcher 실행 후 다운받은 ubuntu iso 파일 선택
5. Select target 으로 내가 이미지를 저장할 USB 선택
6. Flash! 클릭
7. Ubuntu USB 를 서버컴에 연결하고 아래 링크 참고하여 설치
(https://junorionblog.co.kr/ubuntu-20-04-desktop-%EC%84%A4%EC%B9%98-%EA%B0%80%EC%9D%B4%EB%93%9C/)

# 2. Server 에 기본 프로그램 설치
### 2.1.  Git, SSH
```bash
sudo apt install -y vim git openssh-server
```

### 2.2.  Python 설치
```bash
sudo apt-get install python2.7
sudo apt-get install python-pip python-dev python-setuptools
sudo apt-get install python3-pip python3-venv
sudo -H pip3 install --upgrade pip
sudo -H pip2 install --upgrade pip
sudo reboot
```

### 2.3.  키보드 한글 입력 설치
```bash
sudo apt-get update && sudo apt-get install language-selector-gnome gnome-system-tools gnome-tweaks gnome-shell-extensions net-tools
language support 검색→ install/remove languages → Korean 추가
keyboard input method system: IBUS
sudo reboot
```
* Region & language 검색 → Input sources에 한글 추가

### 2.4.  크롬 설치
```bash
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
sudo apt-get update
sudo apt-get install google-chrome-stable
```

### 2.5.  Nvidia-drive 설치
이 부분은 설치된 그래픽 카드에 맞는 드라이버를 체크해보고 설치할 것.

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
```

### 2.6. Docker 설치
1. 기본 Docker 설치
```bash
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
    
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
   
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo docker run hello-world
```

2. Sudo 안쳐도 되도록 설정
```bash
sudo usermod -aG docker [your-user]
```

3. Nvidia-docker install
(밑은 한 커맨드임)
* 저장소 및 GPG 키 설정
```bash
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
* Nvidia-docker install
```bash
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2
```
* docker 서비스 재시작
```bash
$ sudo systemctl restart docker
```

4. docker-compose
```bash
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

$ sudo chmod +x /usr/local/bin/docker-compose
```


# 3. Docker 사용법 및 container 설정 

### 3.1. Docker pull
```bash
$ docker pull [IMAGE]:[TAG]

# example
$ docker pull nvidia/cuda:11.7.0-devel-ubuntu20.04 # cuda toolkit만 포함
$ docker pull nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 # cuda toolkit, cudnn 포함
```

* docker 는 cudnn, cuda, nvcc 등 설정이 마쳐진 image 를 제공한다.
* 위는 이 image 를 pull 로 가져오는 명령어 이다.
* 자신에게 맞는 이미지는 아래 링크에서 찾아보면 된다.
https://hub.docker.com/search?q=

### 3.2. Generate Docker Container

```bash
$ docker container run -it --shm-size=128G --gpus all -d -p [호스트포트]:[컨테이너포트] --name [컨테이너이름] [이미지이름] /bin/bash 
# --shm-size : 공유메모리의 크기를 결정한다. 이 부분을 따로 설정하지 않으면 공유메모리가 너무 작게 잡혀 에러발생 가능성이 높다.
# -d 백그라운드 실행 
# -p 포트 설정 : 도커컨테이너에 [호스트포트]로 접근하면 [컨테이너포트]로 연결해줌(바인딩) 
# --name : 컨테이너 이름 

# example 
$ docker container run -it --shm-size=128G --gpus all -d -p 33446:8888 --name jeahun 4157de9bccb1 /bin/bash
```

### 3.3. Docker Container 접근

* container 에 접근하기 위해선 아래 명령어를 치고 들어가면 된다.

```bash
$ docker exec -it --privileged [Container 이름 혹은 id] /bin/bash
$ docker exec -it --privileged jeahun /bin/bash
```

* docker image 자체에 cuda toolkit 이 포함되어 있음에도 nvcc 명령어가 먹히지 않는 경우가 있다.
* 이에 아래 명령어를 작성하면 해결된다. (nvcc 링크 설정)

```bash
$ sudo nano ~/.bashrc
$ sudo vim ~/.bashrc
```

* 위의 설정 파일이 열리면 아래 실행

```bash
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
```

* 변경 사항 적용

```bash
$ source ~/.bashrc
```

### 3.5. Container 내부에 Anaconda 설치

```sql
# anaconda 설치파일 설치
# 밑의 명령어 2개중 1개만 실행하면 됨
curl -O https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh

# bash 실행
bash Anaconda3-2022.10-Linux-x86_64.sh
# 주의! : 위의 명령어를 실행하면 내가 지금 접속해 있는 유저의 하위에 설치됨
```

* bash 실행 후 yes, no 를 물어보는 게 2번 뜨는데 모두 yes 하면 됨
* 이후 container 를 다시 연결하면 (base)가 떠 있으면 정상적으로 설치된 것이다.
![](img_store/Pasted%20image%2020231227170204.png)

---

# 3. Ubuntu network 관련 설정

### 3.1. SSH 포트 개방
```bash
$ sudo nano /etc/ssh/sshd_config
```
* 위 명령어로 `sshd_config` 파일로 접근
* 아래 Include 부분은 원래 주석처리 되어 있음. 이를 주석해제
* ssh 개방을 원하는 포트를 `port [port 번호]` 로 명시
![](https://i.imgur.com/T0N5IWL.png)
* 저장 후 아래 명령어 실행
```bash
sudo systemctl restart ssh
```

* ==주의==
* 워의 과정은 일반 유저는 적용이 불가능하므로 ssh 관련 설정은 서버관리자가 직접 작업해야 함.