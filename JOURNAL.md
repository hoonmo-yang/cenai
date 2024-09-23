## 2024-08-09
### GPU Error
갑자기 GPU 오류 발생. `langchain.embeddings.HuggingFaceEmbeddings` 모듈을
사용하는데 내부적으로 `GPURunTime` 오류 발생. 오류 문구 내용은 `CUDA_VISIBLE_DEVICES`가
실행하면서 값이 바뀌었다는 내용인데, 구글링을 해보니 PyTorch Community에 관련 내용이 있었음. 시스템 리부팅을 하면 사라진다고 해서 일단 `.bashrc`에 다음의 내용을 추가
```
export CUDA_VISIBLE_DEVICES=0
```
그 다음에 시스템 리부팅함. 이후 이 오류가 해결되었음.

## 2024-08-14
## MLflow local server error
로컬 서버를 돌리기 위해 `gunicorn` 설치. 23.0.0보다 낮은 버전을 깔아야 동작함.
22.0.0 설치함
```shell
$ pip index versions gunicorn  # gunicorn 배포된 모든 버전 확인
$ pip install gunicorn==22.0.0
```
 
MLflow 서버를 구동시킨 후 해당 주소로 web 연결하면 서버가 뻗어 버림.
querystring_parser 모듈을 임포트할 수 없다는 오류가 발생
```shell
$ mlflow ui
```
`pip`로는 인스톨이 안 됨. pip 오류는 발생 안 하는데 해당 모듈이 라이브러리 디렉토리에
없으며 계속 임포트 오류 발생
`mamba install`로 인스톨하니 성공
```shell
$ mamba install querystring_parser # 주의: -가 아닌 _ 
```

## 2024-08-23
## chromadb 오류
chromadb 0.5.5 버전을 설치했더니 계속 `AttributeError: 'Collection' object has no attribute 'model_fields' 오류가 발생
`pip install`은 오류가 발생. 다시 컴파일을 하려고 시도하는 것 같음. `mamba install chromadb==0.5.3`을 수행함.
0.5.3으로 다운그레이드 했더니 잘 동작함


