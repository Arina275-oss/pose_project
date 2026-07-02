import os
import cv2
from tqdm import tqdm

# Пути
GIF_DIR = 'data/raw/exercise_gif/'
OUTPUT_DIR = 'data/processed/frames/'

# Создаём папку для выходных кадров
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Получаем список всех GIF-файлов
gif_files = [f for f in os.listdir(GIF_DIR) if f.endswith('.gif')]

for gif_file in tqdm(gif_files, desc="Обработка GIF"):
    gif_path = os.path.join(GIF_DIR, gif_file)
    # Читаем GIF как видео через OpenCV
    cap = cv2.VideoCapture(gif_path)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Сохраняем кадр как изображение
        frame_name = f"{os.path.splitext(gif_file)[0]}_frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(OUTPUT_DIR, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    print(f"{gif_file}: извлечено {frame_count} кадров")

print("Все кадры успешно извлечены!")
