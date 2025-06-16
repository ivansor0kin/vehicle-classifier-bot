import os
import torch
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
import uvicorn
import io
from PIL import Image
import logging
import numpy as np
import sys
from torchvision import models, transforms
import torch.nn.functional as F

# Добавляем путь к yolov5, чтобы работали внутренние импорты
yolo_path = os.path.join(os.path.dirname(__file__), 'yolov5')
if yolo_path not in sys.path:
    sys.path.insert(0, yolo_path)

# Импортируем необходимые утилиты из yolov5
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# Настраиваем логирование, чтобы видеть подробные ошибки
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vehicle Detection API")

# --- Глобальные переменные для моделей ---
detection_model = None
classifier_model = None
current_detection_model_name = "yolov5_f1.82.pt"

def get_binary_classifier_transform():
    """Возвращает пайплайн трансформаций для бинарного классификатора."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def classify_car(image: Image.Image) -> tuple[bool, float]:
    """
    Классифицирует изображение, возвращая флаг наличия машины и вероятность.
    """
    global classifier_model
    if classifier_model is None:
        return True, 1.0

    try:
        transform = get_binary_classifier_transform()
        tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = classifier_model(tensor)
            # Применяем Softmax для получения вероятностей
            probabilities = F.softmax(output, dim=1)
            # Вероятность класса "car" (индекс 1)
            car_prob = probabilities[0][1].item()
            # Предсказание (индекс 1 - 'car')
            is_car_detected = output.argmax(1).item() == 1
            return is_car_detected, car_prob
            
    except Exception as e:
        logger.error(f"Ошибка при работе бинарного классификатора: {e}")
        return True, 1.0

@app.on_event("startup")
def load_models():
    """Загружает обе модели при старте сервера."""
    global detection_model, classifier_model, current_detection_model_name
    logger.info("Загрузка моделей...")

    # --- Загрузка модели детекции (YOLOv5) ---
    try:
        detection_model_path = f"models/{current_detection_model_name}"
        if not os.path.exists(detection_model_path):
            raise FileNotFoundError(f"Файл модели детекции не найден: {detection_model_path}")

        detection_model = torch.hub.load('./yolov5', 'custom', path=detection_model_path, source='local')
        detection_model.eval()
        logger.info(f"Модель детекции '{current_detection_model_name}' успешно загружена.")
    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке модели детекции: {e}")
        sys.exit(1)

    # --- Загрузка модели классификации (ResNet50) ---
    try:
        classifier_model_path = "binary_classifier/binary_classifier.pt"
        if not os.path.exists(classifier_model_path):
            logger.warning(f"Файл бинарного классификатора не найден: {classifier_model_path}. Пайплайн будет работать без него.")
        else:
            # Загружаем модель целиком, т.к. она была сохранена через torch.save(model)
            classifier_model = torch.load(classifier_model_path, map_location=torch.device('cpu'))
            classifier_model.eval()
            logger.info(f"Бинарный классификатор '{classifier_model_path}' успешно загружен.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке бинарного классификатора: {e}")
        # Не выходим из программы, просто пайплайн будет работать без него


@app.get("/models", summary="Список доступных моделей детекции")
def list_models():
    global current_detection_model_name
    models_dir = "models"
    if not os.path.isdir(models_dir):
        raise HTTPException(status_code=404, detail="Папка 'models' не найдена.")
    
    pt_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    if not pt_files:
        return {"error": "В папке нет моделей (.pt-файлов)."}
        
    return {"models": pt_files, "current": current_detection_model_name}

@app.post("/set", summary="Сменить активную модель детекции")
def set_model(model_param: str = Query(..., alias="model")):
    global detection_model, current_detection_model_name
    model_path = os.path.join("models", model_param)

    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail=f"Файл модели {model_param} не найден.")

    try:
        detection_model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local')
        detection_model.eval()
        current_detection_model_name = model_param
        logger.info(f"Модель успешно сменена на '{model_param}'.")
        return {"status": f"Модель '{model_param}' успешно загружена."}
    except Exception as e:
        logger.error(f"Ошибка при смене модели на '{model_param}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current", summary="Получить имя текущей модели детекции")
def get_current():
    return {"current_model": current_detection_model_name}


@app.post("/predict", summary="Распознать автомобили на изображении")
async def predict(
    image: UploadFile = File(...),
    conf_thres: float = Query(0.25, description="Порог уверенности модели"),
    iou_thres: float = Query(0.45, description="Порог пересечения для NMS")
):

    #1. Чтение и подготовка изображения
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Некорректный файл изображения: {e}")

    #2. Проверка бинарным классификатором
    is_car_detected, car_probability = classify_car(pil_image)
    
    # Готовим базовый ответ
    response = {"car_probability": car_probability, "boxes": []}

    if not is_car_detected:
        logger.info(f"Бинарный классификатор не обнаружил автомобиль (вероятность: {car_probability:.2f}). Пропускаю детекцию.")
        return response # Возвращаем ответ с вероятностью и пустыми рамками

    #3. Предобработка для YOLOv5
    img0 = np.array(pil_image) # Используем numpy-версию для YOLOv5
    # Изменяем размер и добавляем отступы (letterbox)
    img = letterbox(img0, 640, stride=32, auto=True)[0]
    # Конвертируем в формат (канал, высота, ширина)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    # Преобразуем в тензор, нормализуем и добавляем измерение для батча
    img = torch.from_numpy(img).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    #4. Инференс
    try:
        # Модель возвращает "сырые" предсказания
        pred = detection_model(img)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения модели: {e}")

    #5. Постобработка
    # Применяем Non-Maximum Suppression для фильтрации рамок
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    
    boxes = []
    # `pred` - это список детекций (у нас 1 изображение, поэтому берем первый элемент)
    det = pred[0]

    if det is not None and len(det):
        # Масштабируем рамки с размера 640x640 обратно на оригинальный размер img0
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

        # Формируем JSON-ответ
        for *xyxy, conf, cls in det:
            boxes.append({
                "x1": xyxy[0].item(),
                "y1": xyxy[1].item(),
                "x2": xyxy[2].item(),
                "y2": xyxy[3].item(),
                "confidence": conf.item(),
                "class": int(cls.item()),
                "label": detection_model.names[int(cls.item())]
            })

    response["boxes"] = boxes
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)