# 1. Docker Image Pull

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

# 2. Generate Docker Container

```bash
$ docker container run -it --shm-size=128G --gpus all -d -p [호스트포트]:[컨테이너포트] --name [컨테이너이름] [이미지이름] /bin/bash 
# --shm-size : 공유메모리의 크기를 결정한다. 이 부분을 따로 설정하지 않으면 공유메모리가 너무 작게 잡혀 에러발생 가능성이 높다.
# -d 백그라운드 실행 
# -p 포트 설정 : 도커컨테이너에 [호스트포트]로 접근하면 [컨테이너포트]로 연결해줌(바인딩) 
# --name : 컨테이너 이름 

# example 
$ docker container run -it --shm-size=128G --gpus all -d -p 33446:8888 --name jeahun 4157de9bccb1 /bin/bash
```

* 위의 명령어를 실행했을 때 docker ps 명령어 실행 후 아래 사진처럼 나오면 정상적으로 생성된 것이다.
![](img_store/Pasted%20image%2020231227161650.png)

# 3. Docker Container 접근

container 에 접근하기 위해선 아래 명령어를 치고 들어가면 된다.

```bash
$ docker exec -it --privileged [Container 이름 혹은 id] /bin/bash
$ docker exec -it --privileged jeahun /bin/bash
```

아래 사진처럼 root 로 들어가지면 Docker Container 에 접근한 것이다.

![](img_store/Pasted%20image%2020231227162044.png)
# 4. 초기 Container 설정 관련

* docker container 는 기본적으로 root 만 생성되어 있다.
* Root 에서 모든 작업을 진행하기에는 위험이 따른다.(Docker Container 내부라서 메인서버에는 영향을 안주지만 그래도 위험하다)
* 따라서 User 를 생성 후 그 User 에게 Sudo 권한을 부여하는 방식으로 일부 권한을 차단한다.

```bash
# 아래 명령어들은 docker 의 초기 user 인 root 라고 가정한다.

# 1. apt update
apt update

# 2. sudo 설치
apt install sudo

# 4. root 유저 비밀번호 설정
passwd

# 5. user 추가
adduser [계정(유저)이름]
adduser jeahun

# 6. user 에 sudo 권한 추가
usermod -aG sudo [계정(유저)이름]
usermod -aG sudo jeahun

# 7. user 변경
su - jeahun

# 8. curl, wget 설치
sudo apt install curl
sudo apt install wget
```

docker image 자체에 애초에 cuda toolkit 이 포함되어 있음에도 nvcc 명령어가 먹히지 않는 경우가 있다.
이에 아래 명령어를 작성하면 해결된다. (nvcc 링크 설정)

```bash
$ sudo nano ~/.bashrc
$ sudo vim ~/.bashrc
```

위의 설정 파일이 열리면 아래 실행

```bash
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
```

변경 사항 적용

```bash
$ source ~/.bashrc
```

# 5. Container 내부에 Anaconda 설치

 Docker 를 처음 만들었을 때 대부분 root 로 시작하므로 우리가 일반적인 상황이 아니므로 주의가 필요하다.  Root 에서 무언가 설치를 하는 경우 경고가 많이 뜨게 되는데 이를 피하긴 위해선 귀찮더라도 **user 를 하나 만들어주는 게 좋다**. root 에서 anaconda 를 설치하는 경우 `/home/[username]/anaconda3` 와 같은 형태로 설치가 안될수도 있다. 이러한 문제만이 아니더라도 만약 다른 프로그램을 설치한다고 하더라도 보통 사람들은 root 계정으로 설치를 진행하는 경우가 드물기에 구글링을 하는데에도 어려움이 있을 수 있다.

(사실 권한 문제나 여타 다른 문제들이 발생하더라도 Docker 는 컨테이너만 날리면 돼서 크게 걱정하지 않아도 될 것 같다.)

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


# 6. 그 외 설정들

* 이후부터 jupyer 설치 등과 관련된 사항은 아래 링크의 3.3. 부터 보면 된다.
* https://jeahun10717.tistory.com/44