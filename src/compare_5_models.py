
import os
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torchvision

# ---- OpenPose ----
def load_openpose(proto='models/pose_deploy_linevec.prototxt', weights='models/pose_iter_440000.caffemodel'):
    return cv2.dnn.readNetFromCaffe(proto, weights)

def infer_openpose(net, image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0/255, (368,368), (0,0,0), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    points = []
    for i in range(17):
        heatmap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x = (w * point[0]) / 368
        y = (h * point[1]) / 368
        points.append((x, y, conf))
    return np.array(points)

# ---- MoveNet Lightning ----
def load_movenet_lightning():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

def infer_movenet_lightning(model, image):
    input_size = 192
    img = cv2.resize(image, (input_size, input_size))
    # Приводим к int32 (ожидание модели)
    img = img.astype(np.int32)
    img = np.expand_dims(img, axis=0)  # batch dimension
    # Преобразуем в тензор
    tensor = tf.constant(img, dtype=tf.int32)
    # Вызов модели
    results = model.signatures['serving_default'](tensor)
    kpts = results['output_0'].numpy()[0, 0, :, :]  # (17,3)
    return kpts

# ---- MoveNet Thunder ----
def load_movenet_thunder():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

def infer_movenet_thunder(model, image):
    input_size = 256
    img = cv2.resize(image, (input_size, input_size))
    img = img.astype(np.int32)
    img = np.expand_dims(img, axis=0)
    tensor = tf.constant(img, dtype=tf.int32)
    results = model.signatures['serving_default'](tensor)
    kpts = results['output_0'].numpy()[0, 0, :, :]
    return kpts

# ---- Keypoint R-CNN ----
def load_keypoint_rcnn():
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def infer_keypoint_rcnn(model, image):
    img_tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)[0]
    if len(outputs['keypoints']) == 0:
        return np.zeros((17,3))
    kpts = outputs['keypoints'][0].cpu().numpy()
    conf = kpts[:, 2] / 2.0
    return np.column_stack([kpts[:,0], kpts[:,1], conf])

# ---- Основная часть ----
FRAMES_DIR = 'data/processed/frames/'
test_images = [os.path.join(FRAMES_DIR, f) for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')][:10]
if not test_images:
    print("❌ Нет кадров. Запустите extract_frames.py")
    exit()

print("🔄 Загрузка моделей...")

models = {}
models['YOLOv8n'] = YOLO('yolov8n-pose.pt')
models['OpenPose'] = load_openpose()
models['MoveNet_Lightning'] = load_movenet_lightning()
models['KeypointRCNN'] = load_keypoint_rcnn()
models['MoveNet_Thunder'] = load_movenet_thunder()

print(f"✅ Загружено {len(models)} моделей.\n")

# ---- Функция оценки ----
def evaluate(model, name, image_paths):
    total_time = 0
    scores = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        start = time.time()
        if name == 'YOLOv8n':
            res = model(img, verbose=False)
            kpts = res[0].keypoints
            if kpts is not None:
                kpts = kpts.data.cpu().numpy()[0]
                s = kpts[:, 2].tolist()
            else:
                s = []
        elif name == 'OpenPose':
            kpts = infer_openpose(model, img)
            s = kpts[:, 2].tolist()
        elif name == 'MoveNet_Lightning':
            kpts = infer_movenet_lightning(model, img)
            s = kpts[:, 2].tolist()
        elif name == 'KeypointRCNN':
            kpts = infer_keypoint_rcnn(model, img)
            s = kpts[:, 2].tolist()
        elif name == 'MoveNet_Thunder':
            kpts = infer_movenet_thunder(model, img)
            s = kpts[:, 2].tolist()
        else:
            s = []
        elapsed = time.time() - start
        total_time += elapsed
        if s:
            scores.append(sum(s)/len(s))
    avg_time = total_time / len(image_paths)
    avg_score = sum(scores)/len(scores) if scores else 0
    return {'time': avg_time, 'score': avg_score}

# ---- Запуск ----
print("⏳ Запуск сравнения...")
results = {}
for name, model in models.items():
    print(f"  Тестирование {name}...")
    results[name] = evaluate(model, name, test_images)

# ---- ВЫВОД ТАБЛИЦЫ ----
print("\n" + "="*70)
print("📊 СРАВНЕНИЕ 5 АРХИТЕКТУР (YOLO, OpenPose, MoveNet L, KeypointRCNN, MoveNet T)")
print("="*70)
print(f"{'Модель':<20} {'Среднее время (с)':<18} {'Средний score':<12}")
print("-"*70)
for name, data in results.items():
    print(f"{name:<20} {data['time']:<18.3f} {data['score']:<12.3f}")

# ---- Сохранение ----
os.makedirs('runs', exist_ok=True)
with open('runs/comparison_5_models.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✅ Результаты сохранены в runs/comparison_5_models.json")
