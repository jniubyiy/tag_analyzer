# training_byte_predictor.py
"""
Обучение моделей ParNetTag и BytePredictor в две фазы.
Фаза 1: обучается BytePredictor, ParNetTag заморожен.
Фаза 2: обучается ParNetTag, BytePredictor заморожен.
Градиенты накапливаются на CPU и применяются после каждой фазы.
Потеря: NLLLoss (логарифм правдоподобия) для предсказания следующего байта.
Поддерживается масштабирование потерь через LOSS_WEIGHT_PHASE1 и LOSS_WEIGHT_PHASE2.

Вывод после каждого батча:
- byte_acc: доля правильно предсказанных байтов (по всем позициям)
- tag_acc: доля полностью правильно предсказанных тегов (все байты совпали)
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

from model_parnet_tag import ParNetTag
from model_byte_predictor import BytePredictor
import config_training_byte_predictor as cfg

# ----------------------------------------------------------------------
# Device setup
# ----------------------------------------------------------------------
DEVICE_TAG = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_BP = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Devices – ParNetTag: {DEVICE_TAG}, BytePredictor: {DEVICE_BP}")

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class TagDataset(Dataset):
    def __init__(self, root_dir, max_samples=None):
        self.root_dir = root_dir
        self.all_tags = []
        for txt_path in glob.glob(os.path.join(root_dir, "*.txt")):
            with open(txt_path, 'r', encoding='utf-8') as f:
                raw = f.read().strip()
                tags = [t.strip() for t in raw.split(',') if t.strip()] if raw else []
                self.all_tags.extend(tags)
        if max_samples is not None and max_samples > 0:
            self.all_tags = self.all_tags[:max_samples]
        if not self.all_tags:
            raise RuntimeError(f"No tags found in {root_dir}")
        print(f"Total real tags: {len(self.all_tags)}")

    def __len__(self):
        return len(self.all_tags)

    def __getitem__(self, idx):
        return self.all_tags[idx]

# ----------------------------------------------------------------------
# Tag encoding
# ----------------------------------------------------------------------
def encode_tag(tag: str) -> torch.Tensor:
    tag_bytes = tag.encode('utf-8')[:cfg.TAG_BYTE_LEN]
    if len(tag_bytes) < cfg.TAG_BYTE_LEN:
        tag_bytes += bytes([cfg.PAD_BYTE_VALUE] * (cfg.TAG_BYTE_LEN - len(tag_bytes)))
    return torch.tensor(list(tag_bytes), dtype=torch.long)

def encode_tag_float(tag: str) -> torch.Tensor:
    tag_bytes = tag.encode('utf-8')[:cfg.TAG_BYTE_LEN]
    if len(tag_bytes) < cfg.TAG_BYTE_LEN:
        tag_bytes += bytes([cfg.PAD_BYTE_VALUE] * (cfg.TAG_BYTE_LEN - len(tag_bytes)))
    return torch.tensor(list(tag_bytes), dtype=torch.float32)

# ----------------------------------------------------------------------
# Collate function
# ----------------------------------------------------------------------
def collate_teacher_forcing_fn(batch):
    L = cfg.TAG_BYTE_LEN
    B = len(batch)
    byte_indices = [encode_tag(tag) for tag in batch]
    byte_indices_tensor = torch.stack(byte_indices, dim=0)      # [B, L] long
    full_bytes_float = byte_indices_tensor.float()              # [B, L] float

    prefixes = []
    for t in range(L):
        prefix = torch.zeros_like(full_bytes_float)
        if t > 0:
            prefix[:, :t] = full_bytes_float[:, :t]
        prefixes.append(prefix)

    targets = byte_indices_tensor
    return {
        'full_bytes_float': full_bytes_float,
        'prefixes': prefixes,
        'targets': targets
    }

# ----------------------------------------------------------------------
# Gradient helpers
# ----------------------------------------------------------------------
def add_gradients_to_cpu(accum_dict, model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name not in accum_dict:
                accum_dict[name] = param.grad.clone().cpu()
            else:
                accum_dict[name] += param.grad.cpu()
    model.zero_grad()
    return accum_dict

def apply_cpu_gradients(model, cpu_grads):
    for name, param in model.named_parameters():
        if name in cpu_grads:
            param.grad = cpu_grads[name].to(param.device)
        else:
            param.grad = None

# ----------------------------------------------------------------------
# Loss computation with metrics
# ----------------------------------------------------------------------
def compute_loss_and_gradients(parnet_model, predictor_model, batch, 
                               device_tag, device_bp,
                               train_parnet, train_predictor,
                               loss_weight=1.0):
    """
    Выполняет forward, вычисляет суммарный loss по всем позициям,
    умножает на loss_weight, делает один backward,
    возвращает:
        avg_loss_val (float),
        byte_accuracy (float) - доля правильных байтов (по всем позициям),
        tag_accuracy (float)  - доля полностью правильных тегов,
        cpu_grads (dict)
    """
    full_bytes_float = batch['full_bytes_float'].to(device_tag)
    prefixes = batch['prefixes']
    targets = batch['targets'].to(device_bp)          # [B, L] long

    parnet_vec = parnet_model(full_bytes_float)       # [B, cond_dim]

    num_steps = cfg.TAG_BYTE_LEN
    criterion = nn.NLLLoss()
    total_loss = torch.tensor(0.0, device=device_bp)

    # Для подсчёта метрик
    correct_bytes_total = 0
    correct_tags_total = 0
    total_bytes = 0
    total_tags = 0

    # Пройдём по всем позициям, накапливая loss и предсказания
    # Для вычисления точности тегов нужно знать, все ли байты предсказаны верно для каждого примера.
    # Создадим маску правильности для каждого байта
    B = full_bytes_float.shape[0]
    per_byte_correct = torch.zeros(B, num_steps, dtype=torch.bool, device=device_bp)

    for t in range(num_steps):
        prefix_t = prefixes[t].to(device_bp)
        probs = predictor_model(parnet_vec, prefix_t)   # [B, L, 256]
        prob_t = probs[:, t, :]                         # [B, 256]
        log_prob_t = torch.log(prob_t + 1e-8)           # [B, 256]
        target_t = targets[:, t]                        # [B]

        loss = criterion(log_prob_t, target_t)
        total_loss = total_loss + loss

        # Предсказанный класс
        pred_t = torch.argmax(prob_t, dim=1)            # [B]
        correct_byte = (pred_t == target_t)             # [B] bool
        per_byte_correct[:, t] = correct_byte

        correct_bytes_total += correct_byte.sum().item()
        total_bytes += B

    avg_loss_val = total_loss.item() / num_steps

    # Вычисляем точность по байтам
    byte_accuracy = correct_bytes_total / total_bytes if total_bytes > 0 else 0.0

    # Вычисляем точность по тегам (тег считается правильным, если все его байты верны)
    for i in range(B):
        if per_byte_correct[i].all():
            correct_tags_total += 1
    tag_accuracy = correct_tags_total / B if B > 0 else 0.0

    if train_parnet or train_predictor:
        (total_loss * loss_weight).backward()

    cpu_grads = {}
    if train_parnet:
        cpu_grads = add_gradients_to_cpu(cpu_grads, parnet_model)
    if train_predictor:
        cpu_grads = add_gradients_to_cpu(cpu_grads, predictor_model)

    return avg_loss_val, byte_accuracy, tag_accuracy, cpu_grads

# ----------------------------------------------------------------------
# Training epoch (two phases, per-batch logging)
# ----------------------------------------------------------------------
def train_epoch(parnet_model, predictor_model, loader,
                opt_parnet, opt_predictor, device_tag, device_bp):
    parnet_model.train()
    predictor_model.train()
    
    total_loss1 = 0.0
    total_byte_acc1 = 0.0
    total_tag_acc1 = 0.0
    total_loss2 = 0.0
    total_byte_acc2 = 0.0
    total_tag_acc2 = 0.0
    n_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        # ---------- Phase 1: train predictor only ----------
        for p in parnet_model.parameters():
            p.requires_grad = False
        for p in predictor_model.parameters():
            p.requires_grad = True

        loss1, byte_acc1, tag_acc1, cpu_grads_pred = compute_loss_and_gradients(
            parnet_model, predictor_model, batch, device_tag, device_bp,
            train_parnet=False, train_predictor=True,
            loss_weight=cfg.LOSS_WEIGHT_PHASE1
        )
        
        # ---------- Phase 2: train parnet only ----------
        for p in parnet_model.parameters():
            p.requires_grad = True
        for p in predictor_model.parameters():
            p.requires_grad = False

        loss2, byte_acc2, tag_acc2, cpu_grads_parnet = compute_loss_and_gradients(
            parnet_model, predictor_model, batch, device_tag, device_bp,
            train_parnet=True, train_predictor=False,
            loss_weight=cfg.LOSS_WEIGHT_PHASE2
        )
        
        # ---------- Apply gradients ----------
        apply_cpu_gradients(predictor_model, cpu_grads_pred)
        opt_predictor.step()
        opt_predictor.zero_grad()

        apply_cpu_gradients(parnet_model, cpu_grads_parnet)
        opt_parnet.step()
        opt_parnet.zero_grad()

        total_loss1 += loss1
        total_byte_acc1 += byte_acc1
        total_tag_acc1 += tag_acc1
        total_loss2 += loss2
        total_byte_acc2 += byte_acc2
        total_tag_acc2 += tag_acc2

        # Вывод после каждого батча
        print(f"Batch {batch_idx+1:3d}/{n_batches} | "
              f"P1 loss={loss1:.4f} byte_acc={byte_acc1:.4f} tag_acc={tag_acc1:.4f} | "
              f"P2 loss={loss2:.4f} byte_acc={byte_acc2:.4f} tag_acc={tag_acc2:.4f}")

        if cfg.CLEAR_CACHE_EACH_BATCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss1 = total_loss1 / n_batches
    avg_byte_acc1 = total_byte_acc1 / n_batches
    avg_tag_acc1 = total_tag_acc1 / n_batches
    avg_loss2 = total_loss2 / n_batches
    avg_byte_acc2 = total_byte_acc2 / n_batches
    avg_tag_acc2 = total_tag_acc2 / n_batches
    return (avg_loss1, avg_byte_acc1, avg_tag_acc1,
            avg_loss2, avg_byte_acc2, avg_tag_acc2)

# ----------------------------------------------------------------------
# Checkpoint handling
# ----------------------------------------------------------------------
def get_ckpt_path(name, epoch, dir):
    return os.path.join(dir, f"{name}_epoch{epoch}.pth")

def find_latest(name, dir):
    files = glob.glob(os.path.join(dir, f"{name}_epoch*.pth"))
    if not files:
        return None, 0
    def epoch_from_fname(f):
        m = re.search(r'epoch(\d+)', f)
        return int(m.group(1)) if m else -1
    latest = max(files, key=epoch_from_fname)
    return latest, epoch_from_fname(latest)

def clean_old_ckpts(dir, keep=cfg.MAX_CHECKPOINTS):
    for name in ["parnet", "predictor"]:
        files = glob.glob(os.path.join(dir, f"{name}_epoch*.pth"))
        if len(files) <= keep:
            continue
        files.sort(key=lambda f: int(re.search(r'epoch(\d+)', f).group(1)), reverse=True)
        for old in files[keep:]:
            try:
                os.remove(old)
            except OSError:
                pass

def save_ckpt(epoch, parnet_model, opt_parnet, predictor_model, opt_predictor, dir):
    os.makedirs(dir, exist_ok=True)
    torch.save({'epoch': epoch, 'model_state_dict': parnet_model.state_dict(),
                'optimizer_state_dict': opt_parnet.state_dict()},
               get_ckpt_path("parnet", epoch, dir))
    torch.save({'epoch': epoch, 'model_state_dict': predictor_model.state_dict(),
                'optimizer_state_dict': opt_predictor.state_dict()},
               get_ckpt_path("predictor", epoch, dir))
    clean_old_ckpts(dir)

def load_ckpt_if_exists(parnet_model, opt_parnet, predictor_model, opt_predictor, dir):
    loaded_epoch = 0
    for name, model, opt in [("parnet", parnet_model, opt_parnet),
                             ("predictor", predictor_model, opt_predictor)]:
        path, ep = find_latest(name, dir)
        if path:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"Loaded {name} from epoch {ep}")
            if loaded_epoch == 0:
                loaded_epoch = ep
            else:
                assert ep == loaded_epoch, f"Epoch mismatch for {name}"
    return loaded_epoch

# ----------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------
def train():
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    tag_dataset = TagDataset(cfg.DATASET_TXT_DIR, max_samples=cfg.MAX_TAGS)
    train_loader = DataLoader(tag_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_teacher_forcing_fn, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)

    parnet_model = ParNetTag(hidden_dim=cfg.TAG_MODEL_HIDDEN_DIM,
                             num_layers=cfg.TAG_MODEL_NUM_LAYERS).to(DEVICE_TAG)
    predictor_model = BytePredictor(cond_dim=cfg.PARNET_DIM,
                                    byte_len=cfg.TAG_BYTE_LEN,
                                    hidden_dims=cfg.PREDICTOR_HIDDEN_DIMS,
                                    dropout=cfg.DROPOUT).to(DEVICE_BP)

    opt_parnet = optim.Adam(parnet_model.parameters(), lr=cfg.LEARNING_RATE)
    opt_predictor = optim.Adam(predictor_model.parameters(), lr=cfg.LEARNING_RATE)

    start_epoch = load_ckpt_if_exists(parnet_model, opt_parnet,
                                      predictor_model, opt_predictor,
                                      cfg.MODELS_DIR) + 1

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch} / {cfg.NUM_EPOCHS} ---")
        (loss1, byte_acc1, tag_acc1,
         loss2, byte_acc2, tag_acc2) = train_epoch(
            parnet_model, predictor_model, train_loader,
            opt_parnet, opt_predictor,
            DEVICE_TAG, DEVICE_BP
        )
        print(f"Phase1 (predictor) : loss={loss1:.6f}, byte_acc={byte_acc1:.4f}, tag_acc={tag_acc1:.4f}")
        print(f"Phase2 (parnet)    : loss={loss2:.6f}, byte_acc={byte_acc2:.4f}, tag_acc={tag_acc2:.4f}")

        if epoch % cfg.SAVE_EVERY_EPOCHS == 0:
            save_ckpt(epoch, parnet_model, opt_parnet,
                      predictor_model, opt_predictor,
                      cfg.MODELS_DIR)
            print(f"Checkpoint saved at epoch {epoch}")

    print("Training finished.")

if __name__ == "__main__":
    train()