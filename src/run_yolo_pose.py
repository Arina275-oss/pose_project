import os
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO

# Пути
FRAMES_DIR = 'data/processed/frames/'
OUTPUT_DIR = 'data/processed/results_yolo/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Загрузка модели YOLOv8-pose 
model = YOLO('yolov8n-pose.pt') 

# Получаем все кадры
frame_files = [f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')]
if not frame_files:
    print("Нет кадров в папке!")
    exit()

print(f"Найдено {len(frame_files)} кадров. Обработка...")

all_results = {}

for frame_file in tqdm(frame_files, desc="Обработка"):
    frame_path = os.path.join(FRAMES_DIR, frame_file)
    
    # Инференс
    results = model(frame_path, verbose=False)
    result = results[0]
    
    # Извлечение ключевых точек (17 точек в формате COCO)
    if result.keypoints is not None:
        keypoints = result.keypoints.data.cpu().numpy()[0]  # (17, 3) -> x, y, conf
        scores = keypoints[:, 2].tolist()
        coords = keypoints[:, :2].tolist()
        all_results[frame_file] = {
            'keypoints': coords,
            'scores': scores
        }
    else:
        all_results[frame_file] = {'keypoints': [], 'scores': []}
    
    # Сохраняем визуализацию
    vis_img = result.plot()  # встроенная визуализация
    vis_path = os.path.join(OUTPUT_DIR, f"vis_{frame_file}")
    cv2.imwrite(vis_path, vis_img)

# Сохраняем JSON
with open(os.path.join(OUTPUT_DIR, 'all_predictions.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nОбработано {len(all_results)} кадров. Результаты в {OUTPUT_DIR}")
