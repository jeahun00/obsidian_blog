# MANUAL OF CMMI LAB SERVER

* CMMI LAB SERVER 의 User 구조는 아래와 같이 설정한다.
* 메인 규칙은 아래와 같다.
	1. 모든 일반 사용자는 관리자 권한을 가지지 못한다.
	2. 서버 책임자만 admin 이라는 계정을 사용하며 관리자 권한을 가진다.
	3. 일반 사용자는 Docker 를 컨트롤하지 못하며, docker exec 를 통한 본인의 Container 만 접근한다.
	4. 서버 책임자는 Docker 생성, 삭제, 수정등의 권한을 모두 가진다.
![](img_store/Pasted%20image%2020231227145421.png)

# 1. Admin(관리자)

* 관리자는 서버 전체를 관리하는 사용자이며 모든 권한을 가진다.
* 그렇기에 관리자 계정으로 접근 시 시스템 자체를 건드리는 사용에 주의를 기해야 한다.
* 또한 연구실에서 지정한 관리자 이외에는 관리자 계정 접근을 엄격히 금한다.
* 관리자가 담당하는 범위는 아래와 같다.
	1. 서버 네트워크 설정 관리
	2. Global 에 설치되는 Ubuntu Package 관리
	3. Docker 설정 및 Container 관리

### 1.1. Admin User Information

현재 관리자 권한을 가지는 계정은 아래와 같다

* `admin`

관리자 계정의 비밀번호는 연구실 서버 책임자가 관리한다.

### 1.2. useradd and docker permission

* user 를 생성하고 docker permission 부여는 아래 링크에서 확인

https://github.com/CMMILAB/Server_Manual/blob/main/CMMILAB/%5BUbuntu%5D%20User%20Management%20-%20Admin.md

### 1.3. docker

관리자는 일반 user 가 docker 생성 요청을 하였을 때 아래를 따른다.
#### 1.3.1. Docker 설치 관련
* 현재 서버에는 이미 Docker 가 깔려 있어 이 부분은 생략한다.
* 만약 Docker 를 따로 설치해 보고 싶다면 아래 링크를 참고하라.
https://jeahun10717.tistory.com/42
#### 1.3.2. Docker Container 생성
* Docker Container 생성, Container 내부 Anaconda 설치, Container 내부 jupyerlab 설치 등에 관련한 것은 [이 링크](https://github.com/CMMILAB/Server_Manual/blob/main/CMMILAB/Generate%20Docker%20Container.md) 를 참고하라
* Container 생성시 요청한 User 가 Jupyter 에서 작업을 원할 경우 원하는 포트를 받고 그 포트가 점유되어 있는지 확인 후 생성해 주어야 한다.
* 아래 사진은 [이 링크](https://jeahun10717.tistory.com/44) 의 Container 생성 파트이다. 바로 위 문장에서 언급한 포트는 아래 사진에서 `[호스트 포트]` 부분을 만하는 것이다.
![](img_store/Pasted%20image%2020231227154826.png)

https://jeahun10717.tistory.com/44


---

# 2. User(일반사용자)

* 일반 User 의 경우 관리자에게 요청하여 본인의 계정을 생성하도록 한다.
* 계정이 생성되었으면 그 계정으로 ssh 로 접근한다.
**example**
```bash
ssh [username]@[server ip]
ssh hyuna@10.11.41.235
```
* ssh 로 접근한 이후 **모든 실험은 Docker Container 안에서 진행**한다.

### 2.1. User Information

* User 목록은 아래와 같다.
* 관리자가 이 List 를 관리한다.
1. hyuna
2. jeahun

### 2.2. Connect to Docker Container

* ssh 로 서버 메인 터미널에 접근했다면 Docker Container 에 아래처럼 접근하면된다.
* **일반 User 는 모든 활동을 Docker 안에서만 해야 한다.(서버에서 직접 작업하지 말 것)**
* 본인에게 할당된 Docker Container 가 없다면 관리자에게 생성을 요청한다.
* 또한 만약 Jupyer Lab 을 통한 접근을 원할 경우 관리자에게 원하는 포트를 요청하고 관리자는 이 포트가 점유가 되어있는지 확인 후 Container 를 생성해 주어야 한다.

**container 접근 명령어**
```bash
$ docker exec -it --privileged [container 이름 혹은 id] /bin/bash
$ docker exec -it --privileged cudaUbuntu /bin/bash
```