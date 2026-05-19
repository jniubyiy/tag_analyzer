# convert_compressed_parnets_to_images.py
"""
Утилита для визуальной проверки: преобразует сжатые парнеты из датасета в изображения.
Использует decompressor_inference.pt и decoder_inference.pt, загруженные на ОДНО устройство.
Каждый пример обрабатывается полностью последовательно (сначала декомпрессор, потом декодер).
Сохраняет 10 случайных примеров в ./converted_images_test
"""

import os
import random
import torch
from PIL import Image
import config_training_tag_unit as cfg

# Определяем единое устройство для обеих моделей
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Работаем на устройстве: {DEVICE}")

def load_torchscript(path: str) -> torch.jit.ScriptModule:
    """Загружает TorchScript-модель на выбранное устройство."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель не найдена: {path}")
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    return model

def compressed_parnet_to_pil(compressed_parnet: torch.Tensor,
                             decompressor: torch.jit.ScriptModule,
                             decoder: torch.jit.ScriptModule) -> Image.Image:
    """
    Последовательно преобразует сжатый парнет в RGB-изображение.
    Вход: [C, Hc, Wc] (CPU)
    Выход: PIL Image
    """
    with torch.no_grad():
        # 1. Переносим сжатый парнет на общее устройство и добавляем размерность батча
        x = compressed_parnet.unsqueeze(0).to(DEVICE)   # [1, C, Hc, Wc]
        # 2. Декомпрессор → полный парнет [1, 3, H, W]
        full_parnet = decompressor(x)
        # 3. Декодер → RGB-изображение [1, 3, H, W] в диапазоне [-1, 1]
        rgb = decoder(full_parnet)
        # 4. Возвращаем на CPU, убираем батч-измерение
        rgb = rgb.squeeze(0).cpu()
        # Преобразование в PIL
        arr = (rgb.clamp(-1, 1) + 1) / 2 * 255
        arr = arr.permute(1, 2, 0).to(torch.uint8).numpy()
        return Image.fromarray(arr)

def main():
    dataset_dir = cfg.DATASET_DIR_TAG
    decompressor_path = cfg.DECOMPRESSOR_INFERENCE_PATH
    decoder_path = cfg.DECODER_INFERENCE_PATH
    output_dir = "./converted_images_test"

    if not os.path.isdir(dataset_dir):
        print(f"Папка с датасетом не найдена: {dataset_dir}")
        return

    # Загрузка моделей на одно устройство
    print("Загрузка декомпрессора...")
    decompressor = load_torchscript(decompressor_path)
    print("Загрузка декодера...")
    decoder = load_torchscript(decoder_path)

    # Сбор всех .pt файлов
    pt_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
                if f.endswith('.pt')]
    if not pt_files:
        print("В папке датасета нет .pt файлов.")
        return

    num_samples = min(10, len(pt_files))
    selected = random.sample(pt_files, num_samples)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Конвертируем {num_samples} примеров...")
    for file_path in selected:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        compressed = data['parnet_compressed']      # [C, H//2, W//2] на CPU
        # Обработка одного примера: декомпрессор → декодер (последовательно)
        img = compressed_parnet_to_pil(compressed, decompressor, decoder)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(output_dir, f"{base_name}.png")
        img.save(save_path)
        print(f"  {base_name}.png сохранён")

    print(f"Готово. Результаты в папке: {output_dir}")

if __name__ == "__main__":
    main()