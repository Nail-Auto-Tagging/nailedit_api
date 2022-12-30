FROM python:3.10
COPY . .
RUN apt-get update -y
RUN apt-get -y install libgl1-mesa-glx
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 80
CMD ["python", "app.py"]