FROM jumidlej/raspios-buster:tensorflow2.3.1

USER root

# app folders
WORKDIR /usr/src/all-in-one
RUN mkdir -p /var/log/celery/ /var/run/celery/

# python installation

ARG BUILD_ENV=production
ADD requirements ./requirements

RUN pip install --upgrade pip
RUN pip install -r requirements/$BUILD_ENV.txt

# install gpio library

RUN apt install -y python3-gpiozero

# creating user

ARG GROUP_ID
ARG USER_ID
ARG USER_NAME
ADD adduser ./adduser
RUN ./adduser > /dev/null 2>&1

# change the app directory ownership

RUN chown $USER_NAME:$USER_NAME -R /usr/src/all-in-one/
RUN chown $USER_NAME:$USER_NAME /var/run/celery/ && chown $USER_NAME:$USER_NAME /var/log/celery/ 

USER $USER_NAME
ENV PATH="/home/$USER_NAME/.local/bin:${PATH}"

# bootstrap script

ADD start ./start
ADD start_celery ./start_celery