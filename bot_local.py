import logging
import os
import httpx
import cv2
import io
import numpy as np
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    ContextTypes,
    CommandHandler
)

# Загружаем переменные из .env файла
load_dotenv()

# Теперь токен и URL ОБЯЗАТЕЛЬНО должны быть в .env файле или в переменных окружения
TOKEN = os.getenv("TELEGRAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")

if not TOKEN or not SERVER_URL:
    raise ValueError("Необходимо создать .env файл в папке BOT и указать в нем TELEGRAM_TOKEN и SERVER_URL")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

WELCOME_MESSAGE = (
    "Привет! Я бот для обнаружения и классификации автомобилей по типу кузова.\n\n"
    "Я могу определить 9 типов машин:\n"
    "- Sedan (Седан)\n"
    "- SUV (Внедорожник)\n"
    "- Coupe (Купе)\n"
    "- Convertible (Кабриолет)\n"
    "- Hatchback (Хэтчбек)\n"
    "- Minivan (Минивэн)\n"
    "- Van (Автобус)\n"
    "- Truck (Грузовик)\n"
    "- Other (Другой тип транспорта)\n\n"
)

COMMANDS_MESSAGE = (
    "Доступные команды:\n\n"
    "- /models: Показать доступные модели (на сервере)\n"
    "- /set <имя_модели>: Загрузить выбранную модель\n"
    "- /predict: Переключиться в режим обработки фото\n"
    "- /stop: Остановить обработку и сбросить настройки\n"
)

WAITING_MESSAGE = "Ожидаю фотографию автомобиля..."
STOPPED_MESSAGE = "Бот остановлен. Фото не обрабатываются."
NOT_PREDICT_MESSAGE = "Вы не в режиме /predict. Фото не обрабатывается."

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MESSAGE)
    await update.message.reply_text(COMMANDS_MESSAGE)

    try:
        # Используем httpx для асинхронного запроса
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{SERVER_URL}/current")
            resp.raise_for_status()
            data = resp.json()
        current_model = data.get("current_model", "неизвестно")
        await update.message.reply_text(f"Сейчас используется модель: {current_model}")
    except Exception as e:
        await update.message.reply_text(f"Не удалось узнать текущую модель: {e}")

    context.user_data.clear()

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["predict_mode"] = True
    await update.message.reply_text(WAITING_MESSAGE)

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["predict_mode"] = False
    await update.message.reply_text(STOPPED_MESSAGE)

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Используем httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{SERVER_URL}/models")
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            await update.message.reply_text(f"Ошибка: {data['error']}")
            return

        pt_files = data.get("models", [])
        current_model = data.get("current", "неизвестно")

        if not pt_files:
            await update.message.reply_text("На сервере не найдено моделей (.pt-файлов).")
        else:
            msg = "Доступные модели:\n"
            for m in pt_files:
                if m == current_model:
                    msg += f"- {m} (ТЕКУЩАЯ)\n"
                else:
                    msg += f"- {m}\n"
            await update.message.reply_text(msg)
    except httpx.RequestError as e:
        await update.message.reply_text(f"Ошибка при запросе /models:\n{e}")

async def set_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_parts = update.message.text.strip().split()
    if len(text_parts) < 2:
        await update.message.reply_text("Использование: /set <имя_модели.pt>")
        return

    desired_model = text_parts[1]

    try:
        # Используем httpx
        async with httpx.AsyncClient() as client:
            url = f"{SERVER_URL}/set"
            resp = await client.post(url, params={"model": desired_model})
        
        if resp.status_code == 200:
            data = resp.json()
            if "status" in data:
                await update.message.reply_text(data["status"])
            else:
                await update.message.reply_text(f"Ответ сервера: {data}")
        else:
            await update.message.reply_text(f"Ошибка при загрузке модели: {resp.status_code}\n{resp.text}")
    except httpx.RequestError as e:
        await update.message.reply_text(f"Ошибка при запросе /set:\n{e}")

async def call_inference_api(image_bytes: bytes):
    # Отправляем байты, но теперь с указанием имени файла и типа контента
    files = {"image": ("photo.jpg", image_bytes, "image/jpeg")}
    async with httpx.AsyncClient() as client:
        # Увеличиваем таймаут до 120 секунд для надежности
        resp = await client.post(f"{SERVER_URL}/predict", files=files, timeout=120.0)
    resp.raise_for_status()
    return resp.json()

def draw_boxes_opencv(image_bytes: bytes, boxes: list):
    # Работаем с изображением в памяти
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    for box in boxes:
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        confidence = box["confidence"]
        label_text = f'{box["label"]} {confidence:.2f}'

        # Рисуем рамку и подпись с уверенностью
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            label_text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # Немного увеличим шрифт
            (0, 0, 255),
            2
        )

    # Конвертируем обратно в байты для отправки
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        raise ValueError("Не удалось закодировать обработанное изображение.")
    
    return buffer.tobytes()

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("predict_mode", False):
        await update.message.reply_text(NOT_PREDICT_MESSAGE)
        return

    await update.message.reply_text("Фото получено, обрабатываю. Это может занять до минуты...")
    photo = update.message.photo[-1]
    
    # Скачиваем фото в память (в виде байтов)
    photo_file = await photo.get_file()
    image_bytes_io = io.BytesIO()
    await photo_file.download_to_memory(image_bytes_io)
    image_bytes = image_bytes_io.getvalue()

    try:
        result_json = await call_inference_api(image_bytes)
        boxes = result_json.get("boxes", [])
        car_prob = result_json.get("car_probability", 0.0)

        caption_parts = []
        # 1. Показываем вероятность, только если она низкая
        if car_prob < 0.5:
            caption_parts.append(f"Вероятность наличия автомобиля: {car_prob:.2%}")

        if boxes:
            # 2. Новый формат ответа: каждый объект - отдельная строка
            caption_parts.append("Обнаружено:")
            
            # Сортируем по классу, чтобы сгруппировать одинаковые машины
            sorted_boxes = sorted(boxes, key=lambda x: x['label'])
            
            for box in sorted_boxes:
                label = box['label']
                confidence = box['confidence']
                caption_parts.append(f"- {label}: {confidence:.2%}")

            caption = "\n".join(caption_parts)
            annotated_image_bytes = draw_boxes_opencv(image_bytes, boxes)
            await update.message.reply_photo(photo=annotated_image_bytes, caption=caption)
        else:
            # Если рамки не найдены, но вероятность была низкой, покажем ее
            if caption_parts:
                 await update.message.reply_text(f"{caption_parts[0]}\n\nНа фото не найдено автомобилей с достаточной уверенностью.")
            else:
                 await update.message.reply_text("На фото не найдено автомобилей с достаточной уверенностью.")

    except httpx.TimeoutException:
        await update.message.reply_text("Сервер слишком долго обрабатывает запрос. Пожалуйста, попробуйте еще раз через минуту или отправьте изображение меньшего размера.")
    except httpx.RequestError as e:
        await update.message.reply_text(f"Ошибка при запросе к серверу:\n{e}")
    except Exception as e:
        await update.message.reply_text(f"Произошла внутренняя ошибка: {e}")

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("predict", predict_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CommandHandler("set", set_command))

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling()

if __name__ == "__main__":
    main()