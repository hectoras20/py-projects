FROM python:3.8 

WORKDIR /app-docker
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade app-docker/requirements.txt

COPY . /app-docker

CMD bash -c "while true; do sleep 1; done"