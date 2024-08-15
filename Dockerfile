FROM python:3.13.0rc1-bookworm
# FROM python:3.10-slim

WORKDIR /usr/scr/app

COPY requirements.txt .


# Install dependencies to add PPAs
RUN apt-get update && \
    apt-get install -y -qq ffmpeg aria2 libx11-dev && apt clean && \
    apt-get install -y software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install praat-parselmouth==0.4.4
RUN pip install -r requirements.txt


COPY . .

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "serverflask:app"]

# CMD ["python","serverflask.py"]
