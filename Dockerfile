FROM ubuntu:latest

USER root
WORKDIR /app

COPY ./lib /app/lib
COPY ./training_data /app/training_data
COPY ./run.py ./run.py
COPY ./train.py ./train.py
COPY ./req2.txt /app/req2.txt

RUN apt update && apt install -y python3 python3-pip
RUN pip3 install -r req2.txt --break-system-packages

CMD [ "python","train.py" ]



