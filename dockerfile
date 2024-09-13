FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY data/*.csv ./data/*.csv

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        netbase \
        && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . .

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
