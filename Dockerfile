# syntax=docker/dockerfile:1

FROM python:3.9.16-bullseye
WORKDIR /app

RUN echo "#!/bin/bash\n\$@" > /usr/bin/sudo
RUN chmod +x /usr/bin/sudo
RUN apt-get update && apt-get install -y python3-opencv

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
EXPOSE 8501



