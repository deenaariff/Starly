FROM node:7
MAINTAINER Deen Aariff "aariff.deen@gmail.com"
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY package.json /usr/src/app/
COPY auth.json /usr/src/app/
COPY index.js /usr/src/app/
COPY Client/ /usr/src/app/
ENV GOOGLE_APPLICATION_CREDENTIALS=/usr/src/app/auth.json
RUN npm install 
EXPOSE 3000
CMD [ "npm", "start" ]