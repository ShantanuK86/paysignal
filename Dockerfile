FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY model/ ./model/
COPY src/app.py ./src/app.py

EXPOSE 8000
CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8000"]