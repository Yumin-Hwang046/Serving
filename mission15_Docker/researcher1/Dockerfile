# 1. 베이스 이미지: 파이썬 3.10 버전을 가져옵니다. (가벼운 slim 버전 사용)
FROM python:3.10-slim

# 2. 작업 폴더 설정: 도커 컨테이너 내부에 '/app'이라는 폴더를 만들고 여기서 작업합니다.
WORKDIR /app

# 3. 패키지 설치: 요구사항 목록을 먼저 복사하고, 설치를 진행합니다.
# (--no-cache-dir 옵션을 넣으면 설치 찌꺼기를 남기지 않아 도커 용량이 가벼워집니다)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 및 데이터 복사: 내 컴퓨터의 현재 폴더에 있는 모든 것(코드, data 폴더 등)을 도커 내부로 복사합니다.
COPY . .

# 5. 실행 명령어: 도커 컨테이너가 켜질 때 이 파이썬 파일을 실행하라고 명령합니다.
CMD ["python", "train_saved_model.py"]