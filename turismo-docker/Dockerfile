FROM python:3.8-slim-buster

#Labels as key value pair
LABEL Maintainer="Sandro (Khaos Research Group)"

WORKDIR /turismo

RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
    
COPY . .

CMD [ "python", "./sarimax_exog_evaluation.py"]
