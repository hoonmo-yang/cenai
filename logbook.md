# Log Book

## 2024-12-24
### docker postgres 이슈
**문제:**  postgres container를 삭제하고 다시 시작했는데 패스워드가 안 맞는 오류가 발생하며 
host로 DB 엑세스가 안되는 문제 발견

**원인:**  확인 결과 /lib/var/postgresql/data에 예전 데이터와 설정이 남아 있어서 생긴 문제였음

**해결:** volume mapping이 `/var/lib/docker_volumes/postgresql:/var/lib/postgresql`로 되어 있었음.

호스트 디렉토리인 `/var/lib/docker_volumes/postgresql`을 삭제하고 다시 postgres container를 수행하니
이후 제대로 동작함

```shell
$ sudo rm -rf /var/lib/docker_volumes/postgresql
$ docker stop postgres_cenai
$ docker rm postgres_cenai
$ make up
$ psql -h 127.0.0.1 -U cenai -d cenai_db
```

### html to pdf 출력 이슈
**문제:** `weasyprint`가 Naver cloud VM에서 라이브러리 문제를 일으킴. 더 심각한 것은 출력 속도가 너무 느림
그래서 `weasyprint` 버리고 `pdfkit`을 사용함. `pdfkit`으로 html 문서를 pdf 문서로 변환했는데 한글이 출력되지
않은 문제가 발생함

**원인:** 확인 결과 시스템에 한글 폰트가 설치가 안되어서 발생한 문제

**해결:** 시스템에 한글 폰트 설치하고 실행하니 한글이 완벽하게 출력됨. `weasyprint`와 같은 라이브러리 문제도 없으며
출력 속도가 월등히 빨라서 `pdfkit`으로 전환함

```shell
$ sudo apt update
$ sudo apt install wkhtmltopdf  # install wkhtmltopdf package
$ pip install pdfkit  # install pdfkit, which runs based on wkhtmltopdf
$ sudo apt install fonts-nanum fonts-noto-cjk # install hungul fonts for system fonts
```

