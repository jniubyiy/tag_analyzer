# config_training_classifier.py
"""
Конфигурация для обучения классификатора с замороженным теговым энкодером.
Предобученная модель ParNetTag берётся из папки PRETRAINED_PARNET_DIR.
"""

# ======================== ParNetTag (предобученная) =============
TAG_MODEL_HIDDEN_DIM = 256
TAG_MODEL_NUM_LAYERS = 4
TAG_MODEL_DROPOUT = 0.1
COND_DIM = 128
PRETRAINED_PARNET_DIR = "./models_byte_predictor"   # папка с чекпоинтами parnet_epoch*.pth

# ======================== ConditionedParNetClassifier ============
COMPRESSED_CHANNELS = 4
CLASSIFIER_DROPOUT = 0.2

# ======================== Обучение ==============================
BATCH_SIZE = 1
LEARNING_RATE = 0.00001
NUM_EPOCHS = 1000
RANDOM_SEED = 42

# Данные
DATASET_DIR_TAG = "./prepared_dataset_tag_image"
TAG_BYTE_LEN = 128
PAD_BYTE_VALUE = 0
MAX_TRAIN_IMAGES = None
NEGATIVE_SAMPLE_PROB = 0.5

# Чекпоинты (только для классификатора)
MODELS_DIR = "./models_classifier"
MAX_CHECKPOINTS = 5

# Валидация и тестирование
VALIDATION_SPLIT = 10          # количество примеров для валидации (int) или доля (float)
VAL_EVERY_EPOCHS = 1
TEST_EVERY_EPOCHS = 2
NUM_TEST_EXAMPLES = 10
SAVE_EVERY_EPOCHS = 1

# Директории для визуализации
TESTS_DIR = "./test_classifier"
VAL_TESTS_DIR = "./val_classifier"

# Прочее
CLEAR_CACHE_EACH_BATCH = True
NUM_WORKERS = 0

# Пути к моделям для визуализации (декодер и декомпрессор)
DECODER_INFERENCE_PATH = "./models/decoder_inference.pt"
DECOMPRESSOR_INFERENCE_PATH = "./models_compressor/decompressor_inference.pt"