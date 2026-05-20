# training_classifier.py
"""
Обучение классификатора (ConditionedParNetClassifier) с замороженным теговым энкодером.
Предобученная модель ParNetTag загружается из папки models_byte_predictor и не обучается.
Только одна фаза: обучается классификатор.
Задача: бинарная классификация – принадлежит ли тег изображению.
Добавлено тестирование на тренировочной выборке (сохранение примеров).
"""

import os
import re
import glob
import random
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from model_parnet_tag import ParNetTag
from model_conditioned_parnet_classifier import ConditionedParNetClassifier
import config_training_classifier as cfg

# ----------------------------------------------------------------------
# Device setup
# ----------------------------------------------------------------------
DEVICE_TAG = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CLS = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Devices – Tag encoder (frozen): {DEVICE_TAG}, Classifier: {DEVICE_CLS}")

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class ParnetTagPairDataset(Dataset):
    def __init__(self, root_dir, max_images=None):
        self.root_dir = root_dir
        self.pt_files = {}
        self.txt_files = {}
        for pt_path in glob.glob(os.path.join(root_dir, "*.pt")):
            fname = os.path.splitext(os.path.basename(pt_path))[0]
            self.pt_files[fname] = pt_path
        for txt_path in glob.glob(os.path.join(root_dir, "*.txt")):
            fname = os.path.splitext(os.path.basename(txt_path))[0]
            self.txt_files[fname] = txt_path
        common_keys = sorted(set(self.pt_files.keys()) & set(self.txt_files.keys()))
        if max_images is not None and max_images > 0:
            common_keys = common_keys[:max_images]
        self.keys = common_keys
        if not self.keys:
            raise RuntimeError(f"No matching .pt and .txt pairs in {root_dir}")
        self.id_to_idx = {key: i for i, key in enumerate(self.keys)}

        self.parnets = {}
        self.tags_list = {}
        self.global_tags = set()
        for key in self.keys:
            data = torch.load(self.pt_files[key], map_location='cpu', weights_only=False)
            self.parnets[key] = data['parnet_compressed']
            with open(self.txt_files[key], 'r', encoding='utf-8') as f:
                raw = f.read().strip()
                tags = [t.strip() for t in raw.split(',') if t.strip()] if raw else []
            self.tags_list[key] = tags
            self.global_tags.update(tags)

        self.global_tags = sorted(self.global_tags)
        print(f"Global tag vocabulary size: {len(self.global_tags)}")
        if not self.global_tags:
            self.global_tags = ["<dummy_tag>"]
            print("WARNING: No tags found, using dummy tag.")

        self.positive_pairs = []
        for key in self.keys:
            for tag in self.tags_list[key]:
                self.positive_pairs.append((key, tag))
        print(f"Total positive pairs: {len(self.positive_pairs)}")
        self.tags_set = {key: set(self.tags_list[key]) for key in self.keys}

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        key, correct_tag = self.positive_pairs[idx]
        parnet = self.parnets[key]
        tags_of_image = self.tags_list[key]

        if random.random() < getattr(cfg, 'NEGATIVE_SAMPLE_PROB', 0.5):
            available_neg = [t for t in self.global_tags if t not in tags_of_image]
            if available_neg:
                chosen_tag = random.choice(available_neg)
                cond = encode_tag(chosen_tag)
                label = 0.0
                tag_text = chosen_tag
            else:
                cond = torch.randint(0, 256, (cfg.TAG_BYTE_LEN,), dtype=torch.float32)
                label = 0.0
                tag_text = "<random bytes>"
        else:
            cond = encode_tag(correct_tag)
            label = 1.0
            tag_text = correct_tag

        return {
            'compressed_parnet': parnet,
            'cond_bytes': cond,
            'label': label,
            'example_id': key,
            'tag_text': tag_text
        }

def encode_tag(tag: str) -> torch.Tensor:
    tag_bytes = tag.encode('utf-8')[:cfg.TAG_BYTE_LEN]
    if len(tag_bytes) < cfg.TAG_BYTE_LEN:
        tag_bytes += bytes([cfg.PAD_BYTE_VALUE] * (cfg.TAG_BYTE_LEN - len(tag_bytes)))
    return torch.tensor(list(tag_bytes), dtype=torch.float32)

def collate_fn(batch):
    parnets = torch.stack([item['compressed_parnet'] for item in batch], dim=0)
    cond_bytes = torch.stack([item['cond_bytes'] for item in batch], dim=0)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32).unsqueeze(1)
    example_ids = [item['example_id'] for item in batch]
    tag_texts = [item['tag_text'] for item in batch]
    return {
        'parnets': parnets,
        'cond_bytes': cond_bytes,
        'labels': labels,
        'example_ids': example_ids,
        'tag_texts': tag_texts
    }

def get_conditioning(cond_bytes, tag_model):
    with torch.no_grad():
        return tag_model(cond_bytes.to(DEVICE_TAG))

# ----------------------------------------------------------------------
# Loss and metrics
# ----------------------------------------------------------------------
bce_loss = nn.BCELoss()
def accuracy(preds, targets):
    pred_bin = (preds > 0.5).float()
    return (pred_bin == targets).float().mean()

# ----------------------------------------------------------------------
# Загрузка предобученного ParNetTag
# ----------------------------------------------------------------------
def load_pretrained_parnet(model_dir, device):
    files = glob.glob(os.path.join(model_dir, "parnet_epoch*.pth"))
    if not files:
        raise FileNotFoundError(f"No ParNetTag checkpoint found in {model_dir}")
    def epoch_from_fname(f):
        m = re.search(r'epoch(\d+)', f)
        return int(m.group(1)) if m else -1
    latest_file = max(files, key=epoch_from_fname)
    print(f"Loading pretrained ParNetTag from {latest_file}")
    state_dict = torch.load(latest_file, map_location=device, weights_only=False)['model_state_dict']
    model = ParNetTag(hidden_dim=cfg.TAG_MODEL_HIDDEN_DIM,
                      num_layers=cfg.TAG_MODEL_NUM_LAYERS).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

# ----------------------------------------------------------------------
# Train epoch
# ----------------------------------------------------------------------
def train_epoch(cls_model, loader, opt_cls, device_cls, tag_model, device_tag):
    cls_model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        parnets = batch['parnets'].to(device_cls)
        cond_bytes = batch['cond_bytes'].to(device_tag)
        labels = batch['labels'].to(device_cls)

        cond_vec = get_conditioning(cond_bytes, tag_model)
        outputs = cls_model(parnets, cond_vec)
        loss = bce_loss(outputs, labels)
        acc = accuracy(outputs, labels)

        opt_cls.zero_grad()
        loss.backward()
        opt_cls.step()

        total_loss += loss.item()
        total_acc += acc.item()

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{n_batches} | loss={loss.item():.4f} acc={acc.item():.4f}")

        if cfg.CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    return avg_loss, avg_acc

# ----------------------------------------------------------------------
# Validation (with image saving)
# ----------------------------------------------------------------------
def load_decoder(path):
    if not os.path.exists(path):
        return None
    return torch.jit.load(path, map_location=DEVICE_CLS).eval()

def load_decompressor(path):
    if not os.path.exists(path):
        return None
    return torch.jit.load(path, map_location=DEVICE_CLS).eval()

def decode_to_pil(compressed, decompressor, decoder):
    with torch.no_grad():
        x = compressed.unsqueeze(0).to(DEVICE_CLS)
        full = decompressor(x)
        rgb = decoder(full).squeeze(0).cpu()
        arr = (rgb.clamp(-1, 1) + 1) / 2 * 255
        arr = arr.permute(1, 2, 0).to(torch.uint8).numpy()
        return Image.fromarray(arr)

def save_example_images(out_dir, ex, decompressor, decoder, tag_model):
    os.makedirs(out_dir, exist_ok=True)
    img = decode_to_pil(ex['compressed_parnet'], decompressor, decoder)
    img.save(os.path.join(out_dir, "original_decoded.png"))
    with open(os.path.join(out_dir, "metrics.txt"), 'w') as f:
        f.write(f"True label: {ex['label']}\n")
        f.write(f"Predicted prob: {ex['pred']:.6f}\n")
        f.write(f"Predicted class: {1 if ex['pred'] > 0.5 else 0}\n")
        f.write(f"Tag: {ex['tag_text']}\n")

def run_validation(cls_model, val_loader, epoch, models_dir, opt_cls, decoder, decompressor, tag_model):
    cls_model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    examples = []

    with torch.no_grad():
        for batch in val_loader:
            parnets = batch['parnets'].to(DEVICE_CLS)
            cond_bytes = batch['cond_bytes'].to(DEVICE_TAG)
            labels = batch['labels'].to(DEVICE_CLS)
            ids = batch['example_ids']
            tag_texts = batch['tag_texts']

            cond_vec = get_conditioning(cond_bytes, tag_model)
            outputs = cls_model(parnets, cond_vec)

            loss = bce_loss(outputs, labels)
            acc = accuracy(outputs, labels)

            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

            for i in range(parnets.size(0)):
                examples.append({
                    'id': ids[i],
                    'compressed_parnet': parnets[i].cpu(),
                    'tag_text': tag_texts[i],
                    'label': labels[i].item(),
                    'pred': outputs[i].item()
                })

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    print(f"Validation Epoch {epoch}: Loss={avg_loss:.6f}, Acc={avg_acc:.4f}")

    if not examples or decoder is None or decompressor is None:
        cls_model.train()
        return avg_loss, avg_acc

    # Сохраняем временное состояние классификатора
    temp_path = os.path.join(models_dir, f"temp_cls_restore.pt")
    torch.save(cls_model.state_dict(), temp_path)

    del cls_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for ex in examples:
        out_dir = os.path.join(cfg.VAL_TESTS_DIR, f"epoch_{epoch}", f"example_{ex['id']}")
        save_example_images(out_dir, ex, decompressor, decoder, tag_model)

    # Восстанавливаем
    cls_model = ConditionedParNetClassifier(
        compressed_channels=cfg.COMPRESSED_CHANNELS,
        cond_dim=cfg.COND_DIM,
        dropout=cfg.CLASSIFIER_DROPOUT
    ).to(DEVICE_CLS)
    cls_model.load_state_dict(torch.load(temp_path, map_location=DEVICE_CLS))
    os.remove(temp_path)
    cls_model.train()
    return avg_loss, avg_acc

# ----------------------------------------------------------------------
# Tests on train dataset (using full dataset, not Subset)
# ----------------------------------------------------------------------
def collect_test_examples(cls_model, full_dataset, num_examples, tag_model, device_tag, device_cls):
    cls_model.eval()
    positive_pairs = full_dataset.positive_pairs
    indices = random.sample(range(len(positive_pairs)), min(num_examples, len(positive_pairs)))
    examples = []

    for idx in indices:
        key, pos_tag = positive_pairs[idx]
        parnet = full_dataset.parnets[key]
        # Положительный пример
        cond_pos = encode_tag(pos_tag).unsqueeze(0).to(device_tag)
        with torch.no_grad():
            cond_vec = get_conditioning(cond_pos, tag_model)
            out = cls_model(parnet.unsqueeze(0).to(device_cls), cond_vec)
            pred_pos = out.item()
        examples.append({
            'id': key,
            'compressed_parnet': parnet.cpu(),
            'tag_text': pos_tag,
            'label': 1.0,
            'pred': pred_pos
        })

        # Отрицательный пример
        tags_of_image = full_dataset.tags_list[key]
        available_neg = [t for t in full_dataset.global_tags if t not in tags_of_image]
        if available_neg:
            neg_tag = random.choice(available_neg)
            cond_neg = encode_tag(neg_tag).unsqueeze(0).to(device_tag)
            with torch.no_grad():
                cond_vec = get_conditioning(cond_neg, tag_model)
                out = cls_model(parnet.unsqueeze(0).to(device_cls), cond_vec)
                pred_neg = out.item()
            examples.append({
                'id': f"{key}_neg",
                'compressed_parnet': parnet.cpu(),
                'tag_text': neg_tag,
                'label': 0.0,
                'pred': pred_neg
            })
        else:
            rand_bytes = torch.randint(0, 256, (cfg.TAG_BYTE_LEN,), dtype=torch.float32).unsqueeze(0).to(device_tag)
            with torch.no_grad():
                cond_vec = get_conditioning(rand_bytes, tag_model)
                out = cls_model(parnet.unsqueeze(0).to(device_cls), cond_vec)
                pred_neg = out.item()
            examples.append({
                'id': f"{key}_neg",
                'compressed_parnet': parnet.cpu(),
                'tag_text': "(random bytes)",
                'label': 0.0,
                'pred': pred_neg
            })
    return examples

def run_tests(cls_model, full_dataset, epoch, models_dir, opt_cls, decoder, decompressor, tag_model):
    examples = collect_test_examples(cls_model, full_dataset, cfg.NUM_TEST_EXAMPLES, tag_model, DEVICE_TAG, DEVICE_CLS)
    if not examples or decoder is None or decompressor is None:
        return cls_model, opt_cls

    temp_path = os.path.join(models_dir, f"temp_cls_test_restore.pt")
    torch.save(cls_model.state_dict(), temp_path)

    del cls_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for ex in examples:
        out_dir = os.path.join(cfg.TESTS_DIR, f"epoch_{epoch}", f"example_{ex['id']}")
        save_example_images(out_dir, ex, decompressor, decoder, tag_model)

    cls_model = ConditionedParNetClassifier(
        compressed_channels=cfg.COMPRESSED_CHANNELS,
        cond_dim=cfg.COND_DIM,
        dropout=cfg.CLASSIFIER_DROPOUT
    ).to(DEVICE_CLS)
    cls_model.load_state_dict(torch.load(temp_path, map_location=DEVICE_CLS))
    os.remove(temp_path)
    cls_model.train()
    return cls_model, opt_cls

# ----------------------------------------------------------------------
# Checkpoint handling (classifier only)
# ----------------------------------------------------------------------
def get_ckpt_path(epoch, models_dir):
    return os.path.join(models_dir, f"classifier_epoch{epoch}.pth")

def find_latest_checkpoint(models_dir):
    files = glob.glob(os.path.join(models_dir, "classifier_epoch*.pth"))
    if not files:
        return None, 0
    def epoch_from_fname(f):
        m = re.search(r'epoch(\d+)', f)
        return int(m.group(1)) if m else -1
    latest = max(files, key=epoch_from_fname)
    return latest, epoch_from_fname(latest)

def save_checkpoint(epoch, cls_model, opt_cls, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': cls_model.state_dict(),
        'optimizer_state_dict': opt_cls.state_dict(),
    }, get_ckpt_path(epoch, models_dir))
    files = glob.glob(os.path.join(models_dir, "classifier_epoch*.pth"))
    if len(files) > cfg.MAX_CHECKPOINTS:
        files.sort(key=lambda f: int(re.search(r'epoch(\d+)', f).group(1)), reverse=True)
        for old in files[cfg.MAX_CHECKPOINTS:]:
            try:
                os.remove(old)
            except OSError:
                pass

def load_checkpoint_if_exists(cls_model, opt_cls, models_dir):
    path, epoch = find_latest_checkpoint(models_dir)
    if path:
        ckpt = torch.load(path, map_location=DEVICE_CLS, weights_only=False)
        cls_model.load_state_dict(ckpt['model_state_dict'])
        opt_cls.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Loaded classifier from epoch {epoch}")
        return epoch
    return 0

# ----------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------
def train():
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    decoder = load_decoder(cfg.DECODER_INFERENCE_PATH)
    decompressor = load_decompressor(cfg.DECOMPRESSOR_INFERENCE_PATH)
    if decoder is None or decompressor is None:
        print("Warning: decoder/decompressor missing – image saving disabled")

    # Create full dataset (not subset) for testing
    full_dataset = ParnetTagPairDataset(cfg.DATASET_DIR_TAG,
                                        max_images=getattr(cfg, 'MAX_TRAIN_IMAGES', None))
    total_pairs = len(full_dataset)
    if isinstance(cfg.VALIDATION_SPLIT, float):
        val_size = int(total_pairs * cfg.VALIDATION_SPLIT)
    else:
        val_size = min(cfg.VALIDATION_SPLIT, total_pairs)
    train_size = total_pairs - val_size
    if train_size <= 0:
        raise RuntimeError(f"Not enough data: total pairs={total_pairs}, val={val_size}")
    # Random split returns Subset objects; we still keep full_dataset for testing.
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
    )
    print(f"Train pairs: {train_size}, Validation pairs: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
                            pin_memory=True) if val_size > 0 else None

    # Load pretrained frozen ParNetTag
    tag_model = load_pretrained_parnet(cfg.PRETRAINED_PARNET_DIR, DEVICE_TAG)

    # Classifier
    cls_model = ConditionedParNetClassifier(
        compressed_channels=cfg.COMPRESSED_CHANNELS,
        cond_dim=cfg.COND_DIM,
        dropout=cfg.CLASSIFIER_DROPOUT
    ).to(DEVICE_CLS)
    opt_cls = optim.Adam(cls_model.parameters(), lr=cfg.LEARNING_RATE)

    start_epoch = load_checkpoint_if_exists(cls_model, opt_cls, cfg.MODELS_DIR) + 1

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} / {cfg.NUM_EPOCHS} ---")
        loss, acc = train_epoch(cls_model, train_loader, opt_cls, DEVICE_CLS, tag_model, DEVICE_TAG)
        print(f"Train: loss={loss:.6f}, acc={acc:.4f}")

        if val_loader and epoch % cfg.VAL_EVERY_EPOCHS == 0:
            val_loss, val_acc = run_validation(cls_model, val_loader, epoch, cfg.MODELS_DIR,
                                               opt_cls, decoder, decompressor, tag_model)
            print(f"Validation: loss={val_loss:.6f}, acc={val_acc:.4f}")

        if epoch % cfg.TEST_EVERY_EPOCHS == 0:
            print(f"Running tests for epoch {epoch}...")
            cls_model, opt_cls = run_tests(cls_model, full_dataset, epoch, cfg.MODELS_DIR,
                                           opt_cls, decoder, decompressor, tag_model)

        if epoch % cfg.SAVE_EVERY_EPOCHS == 0:
            save_checkpoint(epoch, cls_model, opt_cls, cfg.MODELS_DIR)
            print(f"Checkpoint saved at epoch {epoch}")

    print("Training finished.")

if __name__ == "__main__":
    train()