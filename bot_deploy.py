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
import asyncio

load_dotenv()

#При деплое эти переменные должны быть установлены на сервере
TOKEN = os.getenv("TELEGRAM_TOKEN")
# Порт для внутреннего сервера. Railway устанавливает эту переменную.
PORT = os.getenv("PORT", "5000")
# Если переменная SERVER_URL задана, используем ее.
# Если нет (как в деплое на Railway), собираем URL для localhost с нужным портом.
SERVER_URL = os.getenv("SERVER_URL", f"http://127.0.0.1:{PORT}")

if not TOKEN:
    raise ValueError("Необходимо установить переменную окружения: TELEGRAM_TOKEN")

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
    "- /predict: Переключиться в режим обработки фото\n"
    "- /stop: Остановить обработку и сбросить настройки\n"
)

WAITING_MESSAGE = "Ожидаю фотографию автомобиля..."
STOPPED_MESSAGE = "Бот остановлен. Фото не обрабатываются."
NOT_PREDICT_MESSAGE = "Вы не в режиме /predict. Фото не обрабатывается."

async def wait_for_server(url: str):
    """Ожидает, пока сервер не станет доступен."""
    logger.info("Ожидание запуска FastAPI сервера...")
    # Ожидаем, пока веб-сервер хотя бы начнет отвечать
    for i in range(15):  # Попытаться 15 раз (30 секунд)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health", timeout=2.0)
                if response.status_code == 200:
                    logger.info("Сервер FastAPI доступен! Ожидание загрузки моделей...")
                    # Теперь даем время на загрузку моделей.
                    # Вместо сложной логики, просто ждем фиксированное время.
                    # В реальном проекте здесь бы была проверка на готовность моделей.
                    await asyncio.sleep(15)
                    logger.info("Предполагается, что модели загружены. Бот запускается.")
                    return True
        except httpx.RequestError:
            logger.info(f"Сервер еще не доступен. Попытка {i + 1}/15.")
            await asyncio.sleep(2)
    logger.error("Сервер не запустился за отведенное время.")
    return False

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_MESSAGE)
    await update.message.reply_text(COMMANDS_MESSAGE)

    try:
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

async def call_inference_api(image_bytes: bytes):
    files = {"image": ("photo.jpg", image_bytes, "image/jpeg")}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{SERVER_URL}/predict", files=files, timeout=120.0)
    resp.raise_for_status()
    return resp.json()

def draw_boxes_opencv(image_bytes: bytes, boxes: list):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    for box in boxes:
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        confidence = box["confidence"]
        label_text = f'{box["label"]} {confidence:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            label_text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

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
    
    photo_file = await photo.get_file()
    image_bytes_io = io.BytesIO()
    await photo_file.download_to_memory(image_bytes_io)
    image_bytes = image_bytes_io.getvalue()

    try:
        result_json = await call_inference_api(image_bytes)
        boxes = result_json.get("boxes", [])
        car_prob = result_json.get("car_probability", 0.0)

        caption_parts = []
        if car_prob < 0.5:
            caption_parts.append(f"Вероятность наличия автомобиля: {car_prob:.2%}")

        if boxes:
            caption_parts.append("Обнаружено:")
            
            sorted_boxes = sorted(boxes, key=lambda x: x['label'])
            
            for box in sorted_boxes:
                label = box['label']
                confidence = box['confidence']
                caption_parts.append(f"- {label}: {confidence:.2%}")

            caption = "\n".join(caption_parts)
            annotated_image_bytes = draw_boxes_opencv(image_bytes, boxes)
            await update.message.reply_photo(photo=annotated_image_bytes, caption=caption)
        else:
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
    # Проверяем доступность сервера перед запуском бота
    server_ready = asyncio.run(wait_for_server(SERVER_URL))
    if not server_ready:
        logger.critical("Бот не будет запущен, так как сервер не отвечает.")
        return

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("predict", predict_command))
    app.add_handler(CommandHandler("stop", stop_command))

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling()

if __name__ == "__main__":
    main() 