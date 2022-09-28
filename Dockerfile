FROM python:3.8-slim
MAINTAINER blazh

WORKDIR /usr/src/app
COPY . .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt
RUN apt install -y libsndfile1-dev
EXPOSE 8080
CMD python ./MoeGoeWeb.py