FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src/app.py .
COPY src/model.pth .

EXPOSE 5000

CMD ["python", "app.py"]
