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
1. Создайте виртуальное окружение:
   python -m venv pose_env;
   pose_env\Scripts\activate.
2. Установите все зависимости:
   pip install -r requirements.txt.
3. Скачайте файлы моделей OpenPose (см. раздел «Скачивание моделей»).
4. Запустите извлечение кадров: `python src/extract_frames.py`
5. Запустите инференс: `python src/run_yolo_pose.py`
6. Анализ углов: `python src/analyze_angles.py`
7. Сравнение моделей: `python src/compare_5_models.py`

## Результаты
Лучшая модель: YOLOv8n-pose (скорость 0,084 с/кадр, score 0,637).

## Ссылки
- [Датасет на Kaggle](https://www.kaggle.com/datasets/dilanarvand/exercise-gif-dataset)
  
## Скачивание моделей OpenPose

Для работы с моделью OpenPose необходимо скачать файлы весов и поместить их в папку `models/` (или `openpose-models/pose/coco/`).

### Необходимые файлы

| Файл | Ссылка | Назначение |
|------|--------|------------|
| pose_iter_440000.caffemodel | [Скачать](https://huggingface.co/gaijingeek/openpose-models/resolve/main/models/pose/coco/pose_iter_440000.caffemodel?download=true) | OpenPose COCO (18 точек) |
| pose_iter_584000.caffemodel | [Скачать](https://huggingface.co/gaijingeek/openpose-models/resolve/main/models/pose/body_25/pose_iter_584000.caffemodel?download=true) | OpenPose BODY_25 (25 точек) |
| pose_iter_160000.caffemodel | [Скачать](https://huggingface.co/gaijingeek/openpose-models/resolve/main/models/pose/mpi/pose_iter_160000.caffemodel?download=true) | OpenPose MPI |
| pose_iter_102000.caffemodel | [Скачать](https://huggingface.co/gaijingeek/openpose-models/resolve/main/models/hand/pose_iter_102000.caffemodel?download=true) | OpenPose Hand |

**Важно:** Эти файлы не включены в репозиторий из-за ограничений GitHub на размер файлов (максимум 100 МБ).
