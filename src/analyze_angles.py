import os
import json
import numpy as np

def calculate_angle(a, b, c):
    """Расчёт угла между тремя точками (a-b-c) в градусах"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# ===== ПУТЬ К РЕЗУЛЬТАТАМ YOLO =====
JSON_PATH = 'data/processed/results_yolo/all_predictions.json'

# Проверка существования файла
if not os.path.exists(JSON_PATH):
    print(f"Файл {JSON_PATH} не найден!")
    print("Убедитесь, что вы запустили run_yolo_pose.py и получили результаты.")
    exit()

# Загрузка данных
with open(JSON_PATH, 'r') as f:
    results = json.load(f)

# Индексы ключевых точек COCO (YOLO использует те же):
# 5 - левое бедро, 6 - правое бедро
# 7 - левое колено, 8 - правое колено
# 9 - левая лодыжка, 10 - правая лодыжка
# 11 - левое плечо, 12 - правое плечо
# 13 - левый локоть, 14 - правый локоть
# 15 - левое запястье, 16 - правое запястье

angle_results = {}

for frame_name, data in results.items():
    keypoints = data.get('keypoints', [])
    if len(keypoints) < 17:
        continue
    
    try:
        # Углы в коленях (бедро-колено-лодыжка)
        knee_left = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
        knee_right = calculate_angle(keypoints[6], keypoints[8], keypoints[10])
        
        # Углы в локтях (плечо-локоть-запястье)
        elbow_left = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
        elbow_right = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
        
        angle_results[frame_name] = {
            'knee_angle_left': knee_left,
            'knee_angle_right': knee_right,
            'elbow_angle_left': elbow_left,
            'elbow_angle_right': elbow_right
        }
    except Exception as e:
        print(f"Ошибка на кадре {frame_name}: {e}")
        continue

# Сохраняем углы в отдельный JSON
OUTPUT_JSON = 'data/processed/results_yolo/angles.json'
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(angle_results, f, indent=2)

# Вывод статистики
if angle_results:
    print("\nСТАТИСТИКА УГЛОВ (по всем кадрам):")
    angle_names = ['knee_angle_left', 'knee_angle_right', 
                   'elbow_angle_left', 'elbow_angle_right']
    for name in angle_names:
        values = [v[name] for v in angle_results.values() if name in v]
        if values:
            print(f"  {name}: средний = {np.mean(values):.1f}°, "
                  f"мин = {np.min(values):.1f}°, макс = {np.max(values):.1f}°")
    
    # Анализ техники (пример для правого колена)
    print("\nАНАЛИЗ ТЕХНИКИ (первые 10 кадров):")
    for i, (frame, angles) in enumerate(list(angle_results.items())[:10]):
        knee = angles.get('knee_angle_right', 0)
        if knee > 100:
            status = "Недостаточная глубина"
        elif knee < 70:
            status = "Слишком глубоко (перегрузка)"
        else:
            status = "Оптимальный угол"
        print(f"  {frame}: колено = {knee:.1f}° → {status}")
else:
    print("Нет данных для анализа углов.")

print(f"\nРезультаты углов сохранены в {OUTPUT_JSON}")
