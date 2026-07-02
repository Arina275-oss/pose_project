# Проект по оценке позы человека и анализу техники упражнений

## Описание
Данный проект разработан в рамках практики по системам искусственного интеллекта.  
Цель: сравнение архитектур pose estimation и анализ техники упражнения «выпад».

## Используемые модели
- YOLOv8n-pose
- OpenPose
- MoveNet Lightning
- Keypoint R-CNN
- MoveNet Thunder

## Структура проекта
- `src/` — исходный код (скрипты инференса, анализа углов, сравнения моделей)
- `data/` — данные (извлечённые кадры, результаты)
- `models/` — файлы моделей (локально, не загружены в репозиторий)
- `runs/` — результаты экспериментов в JSON

## Инструкция по запуску
1. Установите зависимости: `pip install -r requirements.txt`
2. Запустите извлечение кадров: `python src/extract_frames.py`
3. Запустите инференс: `python src/run_yolo_pose.py`
4. Анализ углов: `python src/analyze_angles.py`
5. Сравнение моделей: `python src/compare_5_models.py`

## Результаты
Лучшая модель: YOLOv8n-pose (скорость 0,084 с/кадр, score 0,637).

## Ссылки
- [Датасет на Kaggle](https://www.kaggle.com/datasets/dilanarvand/exercise-gif-dataset)
