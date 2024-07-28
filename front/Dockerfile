FROM continuumio/miniconda3

WORKDIR /home/app

RUN apt-get update -y \
    && apt-get install -y \
    nano \
    unzip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -n fastapi-env python=3.8 -y

SHELL ["conda", "run", "-n", "fastapi-env", "/bin/bash", "-c"]

RUN conda install pip -y

RUN conda run -n fastapi-env pip install gunicorn

COPY requirements.txt .

RUN conda run -n fastapi-env pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD gunicorn main:app --bind 0.0.0.0:$PORT --timeout 100 --worker-class uvicorn.workers.UvicornWorker -n fastapi-env