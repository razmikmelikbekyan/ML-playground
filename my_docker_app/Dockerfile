FROM debian:latest

#RUN apt-get update \
#  && apt-get install -y python3-pip python3-dev \
#  && cd /usr/local/bin \
#  && ln -s /usr/bin/python3 python \
#  && pip3 install --upgrade pip

#RUN apt-get autoclean autoremove

#ENTRYPOINT ["python3"]

#ADD my_app.py /

#RUN pip install numpy

#CMD [ "python", "./my_app.py" ]

RUN apt-get update && apt-get install \
  -y build-essential python3.7-dev python3-virtualenv vim

RUN apt-get install -y vim
RUN apt-get autoclean autoremove


ENV VIRTUAL_ENV=/opt/rrp4p-cproxy-venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN which python

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

RUN which python
SHELL ["/bin/bash", "-c"]
RUN echo I am using bash, which is now the default

# Run the application:
COPY my_app.py .
CMD /bin/bash