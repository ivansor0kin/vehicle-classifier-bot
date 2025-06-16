FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    supervisor \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем все файлы проекта
COPY . /app

# Устанавливаем Python зависимости
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Порт для FastAPI сервера
EXPOSE 5000

# Запускаем supervisor
CMD ["/usr/bin/supervisord", "-c", "/app/supervisord.conf"] 