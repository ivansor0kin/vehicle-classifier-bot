FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем все файлы проекта
COPY . /app

# Устанавливаем Python зависимости
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Порт для FastAPI сервера
EXPOSE 5000

# Запускаем supervisor
CMD ["/usr/bin/supervisord", "-c", "/app/supervisord.conf"] 