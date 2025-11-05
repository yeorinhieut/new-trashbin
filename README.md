# 초단간 똥통세팅법

## 준비물
- Docker  
- Docker Compose v2  

## ⚠️ 주의  
config.toml, ./db 폴더가 존재하는지 **반드시 확인하세요!**  
차단을 피하기 위해 **적절한 딜레이 설정**을 권장드립니다.  
(그냥 예제파일을 그대로 따라가는 걸 추천드립니다 😊)

## 빌드 & 실행

```bash

cp -n config.toml.example config.toml

mkdir -p db

docker compose up -d --build
```

접속: http://localhost:8000

## 업데이트

```bash

docker compose down

git pull

docker compose up -d --build --no-cache
```

## 🧠 주주의

99.98퍼센트 llm 작성 코드입니다.
고장나면 수정해주세요 🙃
