# ============================================================
# Snapshot-Teacher + EWC for Task-Incremental Learning
# ============================================================

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# -------------------------------
# Reproducibility
# -------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Enhanced Distillation utilities
# -------------------------------

def temperature_softmax(x, T):
    """Temperature-scaled softmax with numerical stability."""
    x = x / T
    x = x - torch.max(x, dim=1, keepdim=True)[0]  # for numerical stability
    return torch.softmax(x, dim=1)

def kl_divergence(p, q):
    """KL divergence with clipping for stability."""
    q = torch.clamp(q, min=1e-8)
    return torch.sum(p * torch.log(p / q), dim=-1)

def distillation_loss(student_logits, teacher_logits, T=3.0, alpha=0.5):
    """
    Combined distillation loss with temperature scaling.
    Args:
        student_logits: Student model outputs
        teacher_logits: Teacher model outputs
        T: Temperature for softening
        alpha: Weight between hard and soft targets
    """
    # Soft targets
    soft_targets = temperature_softmax(teacher_logits, T)
    soft_prob = temperature_softmax(student_logits, T)

    # KL divergence loss
    kd_loss = kl_divergence(soft_targets, soft_prob).mean() * (T ** 2)

    return kd_loss

# -------------------------------
# Data Preparation
# -------------------------------

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.25,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.25,))
])

fmnist = datasets.FashionMNIST(root="data/", train=True, download=True)

timestep_task_classes = {
    0: ['Pullover', 'Dress'],
    1: ['Trouser', 'Bag'],
    2: ['Ankle boot', 'Sneaker', 'Sandal'],
    3: ['T-shirt/top', 'Coat', 'Shirt'],
}

VAL_FRAC = 0.1
TEST_FRAC = 0.1
BATCH_SIZE = 128

timestep_tasks = {}
task_test_sets = []

for t, class_names in timestep_task_classes.items():
    class_ids = [fmnist.class_to_idx[c] for c in class_names]
    idxs = [i for i, y in enumerate(fmnist.targets) if y in class_ids]

    # Shuffle indices
    np.random.shuffle(idxs)

    images = [fmnist[i][0] for i in idxs]
    labels = [class_ids.index(fmnist[i][1]) for i in idxs]
    task_ids = [t] * len(images)

    train_len = int(len(images) * (1 - VAL_FRAC - TEST_FRAC))
    val_len = int(len(images) * VAL_FRAC)
    test_len = len(images) - train_len - val_len

    # Apply transforms
    train_images = torch.stack([train_transform(img) for img in images[:train_len]])
    val_test_images = torch.stack([test_transform(img) for img in images[train_len:]])

    train_ds = TensorDataset(
        train_images,
        torch.tensor(labels[:train_len]),
        torch.tensor(task_ids[:train_len])
    )

    val_ds = TensorDataset(
        val_test_images[:val_len],
        torch.tensor(labels[train_len:train_len+val_len]),
        torch.tensor(task_ids[train_len:train_len+val_len])
    )

    test_ds = TensorDataset(
        val_test_images[val_len:],
        torch.tensor(labels[train_len+val_len:]),
        torch.tensor(task_ids[train_len+val_len:])
    )

    timestep_tasks[t] = (train_ds, val_ds)
    task_test_sets.append(test_ds)

# -------------------------------
# Enhanced Model Architecture
# -------------------------------

class Conv4Backbone(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout2d(dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(F.max_pool2d(x, 2))
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(F.max_pool2d(x, 2))
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(F.max_pool2d(x, 2))
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        return x.view(x.size(0), -1)

class TaskHead(nn.Module):
    def __init__(self, n_classes, input_dim=256, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

class MultitaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Conv4Backbone()
        self.task_heads = nn.ModuleDict()
        
    def add_task(self, task_id, n_classes):
        head = TaskHead(n_classes).to(next(self.parameters()).device)
        self.task_heads[str(task_id)] = head
        
    def forward(self, x, task_id):
        feats = self.backbone(x)
        return self.task_heads[str(int(task_id))](feats)
    
    def get_shared_params(self):
        """Get parameters from backbone that are shared across tasks."""
        return [p for n, p in self.named_parameters() if 'backbone' in n]

# -------------------------------
# Enhanced Fisher Information with Accumulation
# -------------------------------

def compute_fisher(model, loader, num_samples=500):
    """Compute Fisher Information matrix with sampling."""
    fisher = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            fisher[n] = torch.zeros_like(p)
    
    model.eval()
    samples_count = 0
    
    for x, y, task_ids in loader:
        if samples_count >= num_samples:
            break
            
        x, y = x.to(device), y.to(device)
        task_id = int(task_ids[0])
        
        model.zero_grad()
        logits = model(x, task_id)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        for n, p in model.named_parameters():
            if p.grad is not None and n in fisher:
                fisher[n] += p.grad.detach() ** 2
        
        samples_count += x.size(0)
    
    # Normalize and add small epsilon
    for n in fisher:
        fisher[n] = fisher[n] / samples_count + 1e-8
    
    return fisher

def accumulate_fisher(prev_fisher, new_fisher, decay=0.9):
    """Accumulate Fisher information with exponential decay."""
    if prev_fisher is None:
        return new_fisher
    
    accumulated = {}
    for n in new_fisher:
        if n in prev_fisher:
            accumulated[n] = decay * prev_fisher[n] + (1 - decay) * new_fisher[n]
        else:
            accumulated[n] = new_fisher[n]
    return accumulated

# -------------------------------
# Enhanced Training with Multiple Teachers
# -------------------------------

model = MultitaskModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Hyperparameters with annealing
lambda_kd = 1.0
lambda_ewc = 50.0  # Increased for better regularization
temperature = 3.0
epochs_per_task = 15

# Store all previous teachers and fisher information
teachers = []  # List of all previous teachers
accumulated_fisher = None
accumulated_params = None

# Training history for analysis
training_history = {
    'task_accuracies': [],
    'losses': defaultdict(list)
}

# -------------------------------
# Enhanced Continual Learning Loop
# -------------------------------

for t, (train_ds, val_ds) in timestep_tasks.items():
    print(f"\n=== Training task {t} ({timestep_task_classes[t]}) ===")

    if str(t) not in model.task_heads:
        n_classes = len(torch.unique(train_ds.tensors[1]))
        model.add_task(t, n_classes)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 5

    for epoch in range(epochs_per_task):
        model.train()
        epoch_losses = {'total': [], 'hard': [], 'kd': [], 'ewc': []}

        for x, y, task_ids in tqdm(train_loader, desc=f"Task {t} Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            task_id = int(task_ids[0])

            optimizer.zero_grad()
            logits = model(x, task_id)
            loss_hard = F.cross_entropy(logits, y)

            # ---- Enhanced KD loss from ALL previous teachers ----
            loss_kd = torch.tensor(0.0, device=device)
            if teachers:
                feats = model.backbone(x)
                for old_t, teacher in enumerate(teachers):
                    if str(old_t) in model.task_heads:
                        s_logits = model.task_heads[str(old_t)](feats)
                        with torch.no_grad():
                            t_feats = teacher.backbone(x)
                            t_logits = teacher.task_heads[str(old_t)](t_feats)

                        # Feature distillation + output distillation
                        feature_loss = F.mse_loss(feats, t_feats) * 0.1
                        output_loss = distillation_loss(s_logits, t_logits, temperature)
                        loss_kd += (output_loss + feature_loss) / len(teachers)

            # ---- Enhanced EWC loss with accumulated Fisher ----
            loss_ewc = torch.tensor(0.0, device=device)
            if accumulated_fisher is not None and accumulated_params is not None:
                for n, p in model.named_parameters():
                    if n in accumulated_fisher and 'backbone' in n:  # Only regularize shared params
                        loss_ewc += (accumulated_fisher[n] *
                                   (p - accumulated_params[n]) ** 2).sum()
                loss_ewc = loss_ewc * (lambda_ewc / (t + 1))  # Anneal EWC strength

            # ---- Total loss ----
            total_loss = loss_hard + lambda_kd * loss_kd + loss_ewc
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track losses
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['hard'].append(loss_hard.item())
            epoch_losses['kd'].append(loss_kd.item() if teachers else 0)
            epoch_losses['ewc'].append(loss_ewc.item())

        scheduler.step()

        # Validation
        model.eval()
        val_acc = []
        with torch.no_grad():
            for x, y, task_ids in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x, task_ids[0])
                val_acc.append((preds.argmax(dim=1) == y).float().mean().item())

        val_acc = np.mean(val_acc)
        print(f"Epoch {epoch}: Loss {np.mean(epoch_losses['total']):.4f}, "
              f"Val Acc: {val_acc:.3f}")

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model for this task
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_model_state)
                break

    # Store teacher
    teachers.append(copy.deepcopy(model).to(device))
    teachers[-1].eval()
    for p in teachers[-1].parameters():
        p.requires_grad = False

    # Update accumulated Fisher and parameters
    current_fisher = compute_fisher(model, train_loader)
    accumulated_fisher = accumulate_fisher(accumulated_fisher, current_fisher)
    accumulated_params = {n: p.detach().clone()
                         for n, p in model.named_parameters()
                         if 'backbone' in n}

# -------------------------------
# Enhanced Final Evaluation
# -------------------------------

print("\n" + "="*50)
print("FINAL EVALUATION")
print("="*50)

model.eval()
all_accuracies = []

with torch.no_grad():
    for t, test_ds in enumerate(task_test_sets):
        loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)
        task_correct = 0
        task_total = 0

        for x, y, task_ids in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x, task_ids[0])
            task_correct += (preds.argmax(dim=1) == y).sum().item()
            task_total += y.size(0)

        task_acc = task_correct / task_total
        all_accuracies.append(task_acc)
        print(f"Task {t} ({timestep_task_classes[t]}): Accuracy = {task_acc:.3f}")

print(f"\nAverage Accuracy: {np.mean(all_accuracies):.3f}")
print(f"Std Deviation: {np.std(all_accuracies):.3f}")

# Compute forgetting measure
if len(all_accuracies) > 1:
    forgetting = []
    for i in range(len(all_accuracies) - 1):
        forgetting.append(max(all_accuracies[:i+1]) - all_accuracies[i])
    print(f"Average Forgetting: {np.mean(forgetting):.3f}")
