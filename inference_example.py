# inference_example.py
"""
Демонстрация использования экспортированных TorchScript-моделей:
  - Encoder
  - Decoder
  - Compressor (уровень 1)
  - Decompressor (уровень 1)

Все модели обновлены: промежуточное представление (парнет) не ограничено по диапазону
и гарантированно не содержит нулевых значений.

═══════════════════════════════════════════════════════════════════════════════
ЧТО ТАКОЕ ПАРНЕТ (Parnet)
═══════════════════════════════════════════════════════════════════════════════

Парнет (сокращение от «parameter network» или «промежуточная сеть») — это
многоканальное изображение-представление, которое создаётся энкодером из
обычного RGB-изображения.  В отличие от латентных векторов в классических
автоэнкодерах, парнет сохраняет пространственную структуру (ширину и высоту),
но каждый пиксель описывается не 3 цветовыми каналами, а тремя абстрактными
каналами, не имеющими прямой цветовой интерпретации.

Основные свойства парнета:
  • Форма: [B, 3, H, W] — 3 канала, пространственное разрешение совпадает с
    входным изображением (обычно 512×512).
  • Диапазон значений не ограничен.  Парнет может содержать положительные,
    отрицательные и большие по модулю числа.
  • Ни одно значение не равно ровно 0 (гарантируется добавлением микроскопической
    константы ε = 1e-8 при создании).  Это важно для последующих стадий сжатия
    и обучения без вырожденных нулевых градиентов.
  • Парнет — это детерминированное представление: одно и то же изображение
    всегда даёт одинаковый парнет.
  • Декодер способен восстановить из парнета RGB-изображение, визуально
    близкое к исходному (в диапазоне [-1, 1]).
  • Сжатый парнет (после компрессора) имеет форму [B, C, H/2, W/2], где
    C — сконфигурированное число каналов (обычно 4 или 12), и наследует
    свойства отсутствия нулей и неограниченного диапазона.

Парнет можно рассматривать как «цифровой негатив», в котором информация
об изображении закодирована в многомерном пространственном паттерне.

═══════════════════════════════════════════════════════════════════════════════
ОПИСАНИЕ МОДЕЛЕЙ
═══════════════════════════════════════════════════════════════════════════════

1. Encoder (Энкодер)
   • Назначение: преобразует RGB-изображение в парнет.
   • Вход:  тензор [B, 3, H, W] в диапазоне [-1, 1] (значения пикселей,
            нормализованные от -1 до 1).
   • Выход: тензор [B, 3, H, W] — парнет.  Диапазон не ограничен,
            отсутствуют точные нули.
   • Архитектура: несколько остаточных блоков (ResidualBlock) без
     понижения разрешения, с финальным свёрточным слоем без активации
     (ранее был Tanh, удалён).  Добавляется ε = 1e-8 для защиты от нулей.

2. Decoder (Декодер)
   • Назначение: восстанавливает RGB-изображение из парнета.
   • Вход:  тензор [B, 3, H, W] — парнет с любыми значениями.
   • Выход: тензор [B, 3, H, W] в диапазоне [-1, 1] (благодаря Tanh на выходе).
   • Архитектура: зеркальная энкодеру, но с Tanh в конце.
   • Важно: декодер не требует, чтобы парнет был в каком-то определённом
     диапазоне; он может обрабатывать даже случайный шум.

3. Compressor (Компрессор, уровень 1)
   • Назначение: уменьшает пространственное разрешение парнета в 2 раза,
     увеличивая число каналов (сжатие с сохранением информации).
   • Вход:  тензор [B, 3, H, W] — парнет.
   • Выход: тензор [B, C, H/2, W/2] — сжатый парнет (C задаётся в конфиге,
            обычно 4 или 12).  Диапазон не ограничен, нули отсутствуют.
   • Архитектура: поднимает число каналов до внутренней размерности,
     затем spatial downscale (stride=2) и финальная проекция в C каналов.

4. Decompressor (Декомпрессор, уровень 1)
   • Назначение: восстанавливает полное пространственное разрешение парнета
     из сжатого представления.
   • Вход:  тензор [B, C, H/2, W/2] — сжатый парнет.
   • Выход: тензор [B, 3, H, W] — восстановленный парнет.  Свойства те же:
     диапазон не ограничен, нулей нет.
   • Архитектура: сначала обработка на низком разрешении, затем
     повышение частоты дискретизации (Upsample + свёртка), уменьшение
     числа каналов до 3.

Все четыре модели экспортированы в TorchScript (файлы *_inference.pt)
и являются самодостаточными – не требуют импорта исходной архитектуры.

═══════════════════════════════════════════════════════════════════════════════
ИСПОЛЬЗОВАНИЕ МОДЕЛЕЙ
═══════════════════════════════════════════════════════════════════════════════

Типичные сценарии:

A. Изображение → парнет → изображение (проверка автоэнкодера)
   encoder(image) -> parnet
   decoder(parnet) -> reconstructed_image

B. Сжатие парнета и восстановление (уровень 1)
   compressor(parnet) -> compressed_parnet
   decompressor(compressed_parnet) -> parnet_restored

C. Полный пайплайн: изображение → сжатый парнет → изображение
   encoder(image) -> parnet
   compressor(parnet) -> compressed
   decompressor(compressed) -> parnet_restored
   decoder(parnet_restored) -> final_image

D. Работа с готовыми парнетами (например, из prepared_dataset_parnet)
   parnet = torch.load("something.pt")['parnet']
   image = decoder(parnet)

E. Генерация изображений из случайных парнетов
   random_parnet = torch.randn(1, 3, 512, 512, device='cuda') * 5
   image = decoder(random_parnet)

Все модели ожидают 4-мерный батч (B, C, H, W).  Одиночный образец
следует подавать с размерностью батча 1.

Файлы моделей лежат в ./models/ и ./models_compressor/ и называются
<имя>_inference.pt (например, encoder_inference.pt).

═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_inference_model(path: str) -> torch.jit.ScriptModule:
    """
    Загружает TorchScript-модель из файла и переводит в режим eval.

    Параметры
    ---------
    path : str
        Путь к .pt-файлу, экспортированному через torch.jit.save.

    Возвращает
    -------
    model : torch.jit.ScriptModule
        Готовая к инференсу модель на выбранном устройстве.
    """
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    return model


# ------------------------- Утилиты для работы с изображениями ------------------

def image_to_tensor(pil_image: Image.Image, size: int = 512) -> torch.Tensor:
    """
    Преобразует PIL RGB-изображение в тензор [1, 3, size, size] в диапазоне [-1, 1].

    Параметры
    ---------
    pil_image : PIL.Image.Image
        Входное изображение в формате RGB.
    size : int
        Целевой размер (ширина = высота).  По умолчанию 512.

    Возвращает
    -------
    tensor : torch.Tensor
        Тензор на устройстве DEVICE, форма [1, 3, size, size], значения от -1 до 1.
    """
    img = pil_image.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return t.to(DEVICE)


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """
    Преобразует тензор [1, 3, H, W] в диапазоне [-1, 1] обратно в PIL Image.

    Параметры
    ---------
    t : torch.Tensor
        Тензор с изображением.

    Возвращает
    -------
    img : PIL.Image.Image
        RGB-изображение.
    """
    arr = t.squeeze(0).clamp(-1, 1).cpu()
    arr = ((arr + 1) / 2).permute(1, 2, 0).numpy() * 255
    return Image.fromarray(arr.astype(np.uint8))


def print_tensor_info(name: str, tensor: torch.Tensor):
    """Выводит форму, min, max и число нулей тензора."""
    shape = tuple(tensor.shape)
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    zero_count = (tensor == 0).sum().item()
    print(f"{name}: shape {shape}, min {min_val:.4f}, max {max_val:.4f}, zeros: {zero_count}")


# ═════════════════════════════════════════════════════════════════════════════
# Пример 1 — отдельное использование каждой модели
# ═════════════════════════════════════════════════════════════════════════════

def demo_individual_models():
    print("=" * 70)
    print("1. ИНДИВИДУАЛЬНАЯ РАБОТА МОДЕЛЕЙ")
    print("=" * 70)

    encoder = load_inference_model("./models/encoder_inference.pt")
    decoder = load_inference_model("./models/decoder_inference.pt")
    compressor = load_inference_model("./models_compressor/compressor_inference.pt")
    decompressor = load_inference_model("./models_compressor/decompressor_inference.pt")

    # Создаём тестовое изображение (серый квадрат)
    test_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    x = image_to_tensor(test_img)   # [1, 3, 512, 512] в [-1, 1]

    with torch.no_grad():
        # Энкодер
        parnet = encoder(x)
        print("\nEncoder:")
        print_tensor_info("  input ", x)
        print_tensor_info("  output", parnet)

        # Декодер
        rec_img = decoder(parnet)
        print("\nDecoder:")
        print_tensor_info("  input ", parnet)
        print_tensor_info("  output", rec_img)

        # Компрессор
        comp1 = compressor(parnet)
        print("\nCompressor (Level 1):")
        print_tensor_info("  input ", parnet)
        print_tensor_info("  output", comp1)

        # Декомпрессор
        decomp1 = decompressor(comp1)
        print("\nDecompressor (Level 1):")
        print_tensor_info("  input ", comp1)
        print_tensor_info("  output", decomp1)

    tensor_to_image(rec_img).save("demo_individual_decoder_output.jpg")
    print("\nРезультат декодера сохранён в demo_individual_decoder_output.jpg\n")


# ═════════════════════════════════════════════════════════════════════════════
# Пример 2 — полный пайплайн: изображение → сжатие → восстановление
# ═════════════════════════════════════════════════════════════════════════════

def demo_full_pipeline():
    print("=" * 70)
    print("2. ПОЛНЫЙ ПАЙПЛАЙН: ИЗОБРАЖЕНИЕ → ПАРНЕТ → СЖАТЫЙ ПАРНЕТ → ИЗОБРАЖЕНИЕ")
    print("=" * 70)

    encoder = load_inference_model("./models/encoder_inference.pt")
    decoder = load_inference_model("./models/decoder_inference.pt")
    compressor = load_inference_model("./models_compressor/compressor_inference.pt")
    decompressor = load_inference_model("./models_compressor/decompressor_inference.pt")

    # Попытка загрузить реальное изображение; при неудаче – синтетический квадрат
    try:
        img = Image.open("test.jpg").convert("RGB")
        print("Используется test.jpg")
    except FileNotFoundError:
        print("test.jpg не найден, используется цветной квадрат.")
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))

    x = image_to_tensor(img)

    with torch.no_grad():
        # Прямой проход
        parnet = encoder(x)                         # [1, 3, 512, 512]
        compressed = compressor(parnet)             # [1, C, 256, 256]
        restored_parnet = decompressor(compressed)  # [1, 3, 512, 512]
        reconstructed = decoder(restored_parnet)    # [1, 3, 512, 512]

    tensor_to_image(reconstructed).save("pipeline_reconstructed.jpg")

    print("\nРазмеры тензоров в пайплайне:")
    print(f"  Исходное изображение : {tuple(x.shape)}")
    print(f"  Парнет               : {tuple(parnet.shape)}")
    print(f"  Сжатый парнет (lvl 1): {tuple(compressed.shape)}")
    print(f"  Восст. парнет        : {tuple(restored_parnet.shape)}")
    print(f"  Финальное изображение: {tuple(reconstructed.shape)}")
    print("\nРезультат сохранён в pipeline_reconstructed.jpg\n")


# ═════════════════════════════════════════════════════════════════════════════
# Пример 3 — модульное использование: генерация из случайного парнета
# ═════════════════════════════════════════════════════════════════════════════

def demo_modular_usage():
    print("=" * 70)
    print("3. МОДУЛЬНОЕ ИСПОЛЬЗОВАНИЕ: ДЕКОДИРОВАНИЕ СЛУЧАЙНОГО ПАРНЕТА")
    print("=" * 70)

    encoder = load_inference_model("./models/encoder_inference.pt")
    decoder = load_inference_model("./models/decoder_inference.pt")
    compressor = load_inference_model("./models_compressor/compressor_inference.pt")
    decompressor = load_inference_model("./models_compressor/decompressor_inference.pt")

    # Создаём случайный парнет с большим разбросом
    dummy_parnet = torch.randn(1, 3, 512, 512, device=DEVICE) * 5
    print("\nСлучайный парнет:")
    print_tensor_info("  parnet", dummy_parnet)

    with torch.no_grad():
        # Декодируем случайный парнет в изображение
        generated_img = decoder(dummy_parnet)
        print_tensor_info("  decoded image", generated_img)

        # Также можно сжать случайный парнет
        compressed_random = compressor(dummy_parnet)
        print_tensor_info("  compressed random parnet", compressed_random)

        # И восстановить обратно
        restored_random = decompressor(compressed_random)
        print_tensor_info("  restored random parnet", restored_random)

        # Декодируем восстановленный парнет
        final_random_img = decoder(restored_random)
        print_tensor_info("  final random image", final_random_img)

    tensor_to_image(generated_img).save("demo_random_parnet_decoded.jpg")
    print("\nИзображение из случайного парнета сохранено в demo_random_parnet_decoded.jpg")

    # Дополнительно: обработка готового парнета из файла
    print("\nПример загрузки парнета с диска:")
    try:
        sample = torch.load("prepared_dataset_parnet/0.pt", map_location=DEVICE, weights_only=False)
        parnet_from_disk = sample['parnet'].unsqueeze(0)  # [1, 3, H, W]
        print_tensor_info("  загруженный parnet", parnet_from_disk)
        with torch.no_grad():
            img = decoder(parnet_from_disk)
        tensor_to_image(img).save("from_saved_parnet.jpg")
        print("  изображение сохранено в from_saved_parnet.jpg")
    except FileNotFoundError:
        print("  prepared_dataset_parnet/0.pt не найден, пропускаем загрузку.")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# Дополнительные примеры использования
# ═════════════════════════════════════════════════════════════════════════════

def demo_advanced_usage():
    print("=" * 70)
    print("4. РАСШИРЕННЫЕ СЦЕНАРИИ")
    print("=" * 70)

    encoder = load_inference_model("./models/encoder_inference.pt")
    decoder = load_inference_model("./models/decoder_inference.pt")
    compressor = load_inference_model("./models_compressor/compressor_inference.pt")
    decompressor = load_inference_model("./models_compressor/decompressor_inference.pt")

    # Сценарий А: сравнение исходного изображения и восстановленного после полного цикла
    try:
        img = Image.open("test.jpg").convert("RGB")
    except FileNotFoundError:
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))

    x = image_to_tensor(img)

    with torch.no_grad():
        # Прямой проход
        parnet = encoder(x)
        compressed = compressor(parnet)
        restored_parnet = decompressor(compressed)
        reconstructed = decoder(restored_parnet)

        # Сравнение
        l1_loss = F.l1_loss(reconstructed, x)
        print(f"\nСценарий А: L1-ошибка между оригиналом и реконструкцией: {l1_loss.item():.6f}")

        # Сценарий B: множественное сжатие (двойной проход)
        compressed2 = compressor(restored_parnet)
        restored_parnet2 = decompressor(compressed2)
        reconstructed2 = decoder(restored_parnet2)
        l1_cycle2 = F.l1_loss(reconstructed2, x)
        print(f"Сценарий B: L1 после двух циклов сжатия: {l1_cycle2.item():.6f}")

        # Сценарий C: инференс на батче (несколько изображений)
        batch = torch.cat([x, x], dim=0)  # два одинаковых изображения
        print(f"\nСценарий C: батч из {batch.shape[0]} изображений, форма {tuple(batch.shape)}")
        batch_parnet = encoder(batch)
        batch_compressed = compressor(batch_parnet)
        batch_restored = decompressor(batch_compressed)
        batch_reconstructed = decoder(batch_restored)
        print(f"  Форма батча после всех шагов: {tuple(batch_reconstructed.shape)}")
        print(f"  Средняя L1 ошибка по батчу: {F.l1_loss(batch_reconstructed, batch).item():.6f}")


if __name__ == "__main__":
    demo_individual_models()
    demo_full_pipeline()
    demo_modular_usage()
    demo_advanced_usage()