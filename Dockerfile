FROM ubuntu:latest
LABEL org.opencontainers.image.source="https://github.com/SteinTokvam/crypto-reinforcement-learning-reseach"


USER root
WORKDIR /app

COPY ./lib /app/lib
COPY ./training_data /app/training_data
COPY ./run.py ./run.py
COPY ./train.py ./train.py
COPY ./req3.txt /app/req3.txt

RUN apt update && apt install -y python3 python3-pip
RUN pip3 install -r req3.txt --break-system-packages

CMD [ "python3","train.py" ]