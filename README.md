# 초단간 똥통세팅법

## 준비물
- Docker
- Docker Compose v2

config.toml, ./db 폴더가 존재하는지 잘 확인하세용!
차단을 피하기 위해 적절한 딜레이를 설정하는걸 추천드립니다.
(그냥 예제파일 따라가는걸 추천)

```bash
mkdir -p db
```

## 빌드 & 실행

```bash
docker compose up -d --build
```

http://localhost:8000.
