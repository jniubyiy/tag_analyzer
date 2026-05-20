# config_training_byte_predictor.py
"""
Конфигурация для обучения ParNetTag и BytePredictor (авторегрессионное предсказание байтов).
"""

# ======================== ParNetTag =============================
TAG_MODEL_HIDDEN_DIM = 256
TAG_MODEL_NUM_LAYERS = 4
TAG_MODEL_DROPOUT = 0.1
PARNET_DIM = 128

# ======================== BytePredictor =========================
PREDICTOR_HIDDEN_DIMS = [256, 256, 128]   # скрытые слои MLP
DROPOUT = 0.2

# ======================== Данные ================================
DATASET_TXT_DIR = "./prepared_dataset_tag_image"   # папка с .txt файлами
MAX_TAGS = None                                    # ограничить количество тегов
TAG_BYTE_LEN = 128
PAD_BYTE_VALUE = 0

# ======================== Обучение ==============================
BATCH_SIZE = 256
LEARNING_RATE = 0.00001
NUM_EPOCHS = 1000
RANDOM_SEED = 42

# Чекпоинты
MODELS_DIR = "./models_byte_predictor"
MAX_CHECKPOINTS = 10
SAVE_EVERY_EPOCHS = 1

# Прочее
CLEAR_CACHE_EACH_BATCH = True
NUM_WORKERS = 0

# ======================== Веса потерь ===========================
LOSS_WEIGHT_PHASE1 = 100.0   # коэффициент для потерь при обучении BytePredictor
LOSS_WEIGHT_PHASE2 = 100.0   # коэффициент для потерь при обучении ParNetTag