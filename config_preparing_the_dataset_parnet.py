# config_preparing_the_dataset_parnet.py

"""
Конфигурация для подготовки датасета парнетов.
Все пути и параметры задаются здесь.
"""

# Целевое разрешение (сторона квадрата). Должно быть кратно 32.
TARGET_RESOLUTION = 512  # 256, 512, 1024, 2048 и т.д.

# Директории
DATASET_DIR = "./dataset_tag_image"                     # исходные изображения
OUTPUT_DIR = "./prepared_dataset_tag_image"    # куда сохранять парнеты (.pt)

# Путь к предобученной инференс-модели энкодера (TorchScript)
ENCODER_MODEL_PATH = "./models/encoder_inference.pt"

# Путь к предобученной инференс-модели компрессора (TorchScript)
COMPRESSOR_MODEL_PATH = "./models_compressor/compressor_inference.pt"

# Количество параллельных процессов (по умолчанию – число ядер CPU)
NUM_WORKERS = 1