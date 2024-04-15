여기서는 ubuntu 에 user 를 생성하고 관리하는 방법과 해당 계정의 네트워크 설정을 다루는 방법을 기술한다.

# 1. User Management

현재 서버의 관리자는 Ubuntu 라고 가정하고 나머지 계정들은 User 로 간주한다.
### 1.1. user 추가

user 를 추가하는 명령어이다.

* **명령어**
```bash
sudo adduser [생성될 계정 아이디]
```
* **예시**
```bash
sudo adduser jeahun
```

![](../img_store/Pasted%20image%2020231227124306.png)

* 여기서 여러가지 정보를 입력하도록 요구한다.
* 이 항목들은 user 개인의 정보들을 요구한다. 많은 글들에서 정확히 안적어도 된다고 하지만 연구실은 철저히 관리해야 하므로 연구실 책임자는 책임지고 이 부분을 챙겨야 한다.

1. `Enter new UNIX password:`
	* password 입력
2. `Retype new UNIX password:`
	* password 재입력
3. `Full Name []: `
	* 본인 전체 이름 입력
4. `Room Number []:` 
	* 연구실 방 번호
5. `Work Phone []:`
	* 개인 전화번호 입력
6. `Home Phone []:`
	* 연구실 전화번호 입력
7. `Other []:`
	* 추가정보 기입

### 1.2. user 삭제

user 를 삭제하고 싶을 때 사용한다.
이 때 sudo 권한을 가지고 있는 계정에서만 해당 명령어 사용이 가능하다.

* **명령어**
```bash
sudo deluser [Ubuntu 계정 아이디]
```
* **예시**
```bash
sudo deluser jeahun
```

아래 창이 뜨면 성공한 것이다.
![](../img_store/Pasted%20image%2020231227131850.png)

### 1.3. user 에게 관리자 권한 부여

usermod 명령어는 특정 그룹에 username 에 해당하는 user 를 포함시키겠다는 명령어이다.
우리는 관리자 그룹에 user 를 추가하는 방법을 알아볼 것이다.

* **명령어**
```bash
usermod -aG sudo [username]
```
* **예시**
```bash
usermod -aG sudo jeahun
```

### 1.4. user 전환

만약 현재 상태가 관리자 user 일 때 다른 계정에 접속해야 하는 상황이 있을 수 있다.
이 때 다른 user 로 변경하는 방법을 알아보자.

* **명령어**
```bash
su - [username]
```
username 으로 계정 변경
* **예시**
```bash
su - jeahun
```

![](../img_store/Pasted%20image%2020231227150109.png)

위 사진처럼 user 이름이 변경되면 성공한 것.


# 2. Docker Permission

### 2.1. Generate Ubuntu Docker Group

```bash
getent group docker
```

* 위의 명령어는 docker group 이 존재하는지 확인하고 없으면 생성하는 명령어
* 처음 한번만 실행하면 된다

### 2.2. Adduser to Docker Group

* 원래 Docker 명령어는 sudo 를 포함하여야 한다.
* 하지만 모든 user 에게 sudo 권한을 주는 것은 보안상 매우 위험하다.
* 이에 docker 관련 명령어는 사용할 수 있도록 부여한다.
* **하지만 일반 유저는 exec 외에는 다른 docker 명령어 사용을 엄격히 금한다.**

* **명령어**
```bash
sudo usermod -aG docker [username]
```
* **사용예**
```bash
sudo usermod -aG docker jeahun
```

* username 에 해당하는 user 에게 docker 명령어 사용권한 부여

### 2.3. Check Docker Group

* 특정 user 가 docker group 에 추가되었는지 확인

```bash
groups [username]
```

* 아래 사진처럼 docker 가 포함되어 뜨면 추가된 거

![](../img_store/Pasted%20image%2020231227152851.png)

* 이제 docker ps 를 써 보면 아래와 같이 실행이 가능함을 알 수 있다.

![](../img_store/Pasted%20image%2020231227153037.png)
