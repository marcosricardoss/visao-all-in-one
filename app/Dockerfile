FROM python:3.8

# app folders
WORKDIR /usr/src/all-in-one
RUN mkdir -p /var/log/celery/ /var/run/celery/

# python installation

ARG BUILD_ENV=production
ADD requirements ./requirements

RUN pip install --upgrade pip
RUN pip install -r requirements/$BUILD_ENV.txt

# creating user

ARG GROUP_ID
ARG USER_ID
ARG USER_NAME

RUN addgroup --gid $GROUP_ID $USER_NAME
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME  

# change the app directory ownership

RUN chown $USER_NAME:$USER_NAME -R /usr/src/all-in-one/
RUN chown $USER_NAME:$USER_NAME /var/run/celery/ && chown $USER_NAME:$USER_NAME /var/log/celery/ 

USER $USER_NAME
ENV PATH="/home/$USER_NAME/.local/bin:${PATH}"

# bootstrap script

ADD start ./start
ADD start_celery ./start_celery