# preparing_the_dataset_parnet.py
import os
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import concurrent.futures
import config_preparing_the_dataset_parnet as cfg

def load_model(model_path: str, device: str = "cpu") -> torch.jit.ScriptModule:
    """Загружает TorchScript-модель из файла."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def process_single_image(args_tuple):
    """
    Обрабатывает одно изображение: открывает, масштабирует, центрирует,
    прогоняет через энкодер, затем через компрессор.
    Сохраняет .pt файл, содержащий ТОЛЬКО сжатый парнет.

    Сохраняемый словарь:
        - 'parnet_compressed' : [C, H//2, W//2]   сжатый парнет

    Принимает кортеж: (file_path_str, target_resolution, output_dir_str,
                       encoder_model_path, compressor_model_path)
    """
    file_path_str, target_resolution, output_dir_str, encoder_model_path, compressor_model_path = args_tuple
    file_path = Path(file_path_str)
    output_path = Path(output_dir_str)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем модели один раз для процесса
    try:
        encoder = load_model(encoder_model_path, device)
    except Exception as e:
        return (file_path.name, f"ERROR: failed to load encoder: {e}")

    try:
        compressor = load_model(compressor_model_path, device)
    except Exception as e:
        return (file_path.name, f"ERROR: failed to load compressor: {e}")

    try:
        # Открываем изображение
        img = Image.open(file_path).convert("RGB")
        w, h = img.size
        max_side = max(w, h)

        # Масштабирование, если нужно
        if max_side > target_resolution:
            ratio = target_resolution / max_side
            new_w = int(round(w * ratio))
            new_h = int(round(h * ratio))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h

        # Центрирование с чёрным паддингом
        canvas = Image.new("RGB", (target_resolution, target_resolution), (0, 0, 0))
        offset_x = (target_resolution - w) // 2
        offset_y = (target_resolution - h) // 2
        canvas.paste(img, (offset_x, offset_y))

        # Бинарная маска (вычисляется, но не сохраняется)
        mask = np.zeros((target_resolution, target_resolution), dtype=np.uint8)
        mask[offset_y:offset_y + h, offset_x:offset_x + w] = 1

        # Преобразование в тензор [0, 1]
        img_tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        # Нормализация в [-1, 1]
        img_tensor = img_tensor * 2.0 - 1.0
        img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

        # Прогон через энкодер
        with torch.no_grad():
            parnet_tensor = encoder(img_tensor)  # [1, 3, H, W]

        # Прогон через компрессор
        with torch.no_grad():
            compressed_tensor = compressor(parnet_tensor)  # [1, C, H//2, W//2]

        # Перенос на CPU, убираем batch-измерение
        compressed_tensor = compressed_tensor.squeeze(0).cpu()  # [C, H//2, W//2]

        # Сохранение ТОЛЬКО сжатого парнета
        save_path = output_path / f"{file_path.stem}.pt"
        torch.save({"parnet_compressed": compressed_tensor}, save_path)

        return (file_path.name, "OK")
    except Exception as e:
        return (file_path.name, f"ERROR: {e}")

def prepare_dataset_parnet(
    target_resolution: int,
    dataset_dir: str,
    output_dir: str,
    encoder_model_path: str,
    compressor_model_path: str
):
    """
    Параллельно обрабатывает изображения, конвертируя их в сжатые парнеты.
    Принимает любые имена файлов.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg'}
    file_paths = []
    for f in sorted(dataset_path.iterdir(), key=lambda x: x.stem):
        if f.suffix.lower() in image_extensions:
            file_paths.append(f)

    if not file_paths:
        print("Нет подходящих изображений в папке dataset.")
        return

    print(f"Найдено {len(file_paths)} изображений. Целевое разрешение: {target_resolution}")
    print(f"Запуск параллельной обработки в {cfg.NUM_WORKERS} процессов...")

    tasks = [(str(p), target_resolution, str(output_path),
              encoder_model_path, compressor_model_path) for p in file_paths]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_image, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            fname = Path(futures[future]).name
            try:
                result = future.result()
                if result[1] == "OK":
                    print(f"OK: {result[0]}")
                else:
                    print(f"FAIL: {result[0]} – {result[1]}")
            except Exception as e:
                print(f"FAIL: {fname} – исключение в процессе: {e}")

    print(f"Готово. Сжатые парнеты сохранены в {output_path}")

if __name__ == "__main__":
    if cfg.TARGET_RESOLUTION % 32 != 0:
        print("Предупреждение: разрешение не кратно 32, архитектура рассчитана на кратность 32.")
    prepare_dataset_parnet(
        target_resolution=cfg.TARGET_RESOLUTION,
        dataset_dir=cfg.DATASET_DIR,
        output_dir=cfg.OUTPUT_DIR,
        encoder_model_path=cfg.ENCODER_MODEL_PATH,
        compressor_model_path=cfg.COMPRESSOR_MODEL_PATH
    )