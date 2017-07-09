FROM ubuntu:latest
MAINTAINER Deen Aariff "aariff.deen@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
ADD yelp_labelled.txt ./
RUN pip install -r requirements.txt
RUN python -m nltk.downloader all-corpora
RUN python -m nltk.downloader third-party
RUN python -m nltk.downloader popular
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["predict.py"]
