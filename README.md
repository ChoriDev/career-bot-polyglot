1. FastAPI와 uvicorn 설치

``` shell
pip install fastapi uvicorn
```

2. KoAlpaca-Polyglot-5.8B 모델 실행을 위한 패키지 설치

``` shell
pip install -U bitsandbytes
pip install -U git+https://github.com/huggingface/peft.git
```

3. 웹 서버 실행

``` shell
uvicorn main:app --host 0.0.0.0 --port 8002
```