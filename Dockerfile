# specify the parent base image which is the python version 3.9-alpine for lightweight
FROM python:3.9

MAINTAINER Ennio Maldonado <techno1731@gmail.com>

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir --upgrade pip

# set work directory
WORKDIR /src

# copy requirements.txt
COPY ./requirements.txt /src/requirements.txt

# install project requirements
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# set work directory
WORKDIR /src

# set app port
EXPOSE 8080

ENTRYPOINT [ "python" ] 

# Run app.py when the container launches
CMD [ "run.py","run","--host","0.0.0.0"] 
