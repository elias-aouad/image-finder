FROM python:3.8

WORKDIR /image-finder

COPY . .

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install -r dev.requirements.txt

RUN chmod +x /image-finder/download_features.sh

CMD ["/bin/bash", "-c", "/image-finder/download_features.sh && python /image-finder/app.py"]

EXPOSE 5000
