# ============================================================
# STANDARD DER++ with 10 examples/task (Minimal Memory)
# Fashion-MNIST Task-Incremental Learning
# ============================================================

import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration
# -------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters (Standard DER++ parameters)
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20
LR = 5e-4
WEIGHT_DECAY = 1e-4
MEMORY_PER_TASK = 15  # Only 10 examples per task!

# Standard DER++ hyperparameters
ALPHA = 0.5  # Weight for logit matching loss (from original DER++ paper)
BETA = 1.0   # Weight for replay classification loss

# -------------------------------
# Data Preparation
# -------------------------------
def prepare_task_incremental_data():
    """Prepare Fashion-MNIST data with task splits."""
    # Data transforms
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
    
    # Load Fashion-MNIST
    fmnist = datasets.FashionMNIST(root="data/", train=True, download=True)
    
    # Task definitions (same as LwF practical)
    timestep_task_classes = {
        0: ['Pullover', 'Dress'],
        1: ['Trouser', 'Bag'],
        2: ['Ankle boot', 'Sneaker', 'Sandal'],
        3: ['T-shirt/top', 'Coat', 'Shirt'],
    }
    
    VAL_FRAC = 0.1
    TEST_FRAC = 0.1
    
    timestep_tasks = {}
    task_test_sets = []
    task_num_classes = []
    
    for t, class_names in timestep_task_classes.items():
        # Get class indices
        class_ids = [fmnist.class_to_idx[c] for c in class_names]
        
        # Get indices for these classes
        idxs = [i for i, y in enumerate(fmnist.targets) if y in class_ids]
        np.random.shuffle(idxs)
        
        # Prepare data
        images = [fmnist[i][0] for i in idxs]
        labels = [class_ids.index(fmnist[i][1]) for i in idxs]
        task_ids = [t] * len(images)
        
        # Split sizes
        train_len = int(len(images) * (1 - VAL_FRAC - TEST_FRAC))
        val_len = int(len(images) * VAL_FRAC)
        test_len = len(images) - train_len - val_len
        
        # Apply transforms
        train_images = torch.stack([train_transform(img) for img in images[:train_len]])
        val_test_images = torch.stack([test_transform(img) for img in images[train_len:]])
        
        # Create datasets
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
        
        # Store tasks
        timestep_tasks[t] = (train_ds, val_ds)
        task_test_sets.append(test_ds)
        task_num_classes.append(len(class_names))
        
        print(f"Task {t} ({class_names}): {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test samples")
    
    return timestep_tasks, task_test_sets, task_num_classes

# -------------------------------
# Model Architecture (Unchanged)
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

# -------------------------------
# Replay Buffer with Importance Sampling (Kept as is)
# -------------------------------
class DERBuffer:
    def __init__(self, capacity_per_task):
        self.capacity_per_task = capacity_per_task
        self.buffer = defaultdict(list)  # task_id -> list of (x, y, logits)
        self.task_logit_shapes = {}  # Store logit shape for each task
        
    def compute_sample_importance(self, model, x, y, task_id):
        """Compute importance score for a sample."""
        # Temporarily switch to eval mode for batch norm
        training_mode = model.training
        model.eval()
        
        with torch.no_grad():
            logits = model(x.unsqueeze(0), task_id)
            # Importance = confidence + feature diversity score
            confidence = F.softmax(logits, dim=1).max().item()
            
            # Feature magnitude as diversity proxy
            features = model.backbone(x.unsqueeze(0))
            diversity = torch.norm(features).item()
            
            # Combined importance score
            importance = confidence * 0.7 + diversity * 0.3
        
        # Restore original mode
        if training_mode:
            model.train()
        
        return importance
    
    def add_samples(self, model, dataset, task_id, num_samples=None):
        """Add the most important samples from a dataset to the buffer."""
        if num_samples is None:
            num_samples = self.capacity_per_task
        
        if len(dataset) == 0:
            return
        
        # Create a loader for the dataset
        loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=False)
        
        all_samples = []
        
        # Use eval mode for importance computation to avoid batch norm issues
        training_mode = model.training
        model.eval()
        
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, task_id)
                
                # Store the logit shape for this task
                if task_id not in self.task_logit_shapes:
                    self.task_logit_shapes[task_id] = logits.shape[1]
                
                # Compute features for diversity
                features = model.backbone(x)
                
                # Store with importance score
                for i in range(len(x)):
                    diversity = torch.norm(features[i:i+1]).item()
                    confidence = F.softmax(logits[i:i+1], dim=1).max().item()
                    importance = confidence * 0.7 + diversity * 0.3
                    
                    all_samples.append({
                        'x': x[i].cpu(),
                        'y': y[i].cpu(),
                        'logits': logits[i].cpu(),
                        'importance': importance
                    })
        
        # Restore original mode
        if training_mode:
            model.train()
        
        # Sort by importance and keep top samples
        all_samples.sort(key=lambda s: s['importance'], reverse=True)
        samples_to_keep = all_samples[:num_samples]
        
        # Clear old samples for this task
        self.buffer[task_id] = []
        
        # Add new samples
        for sample in samples_to_keep:
            self.buffer[task_id].append(
                (sample['x'], sample['y'], sample['logits'])
            )
        
        print(f"  Buffer: Added {len(samples_to_keep)} samples for task {task_id}")
    
    def get_all_samples(self):
        """Get all samples from all tasks."""
        all_samples = []
        for task_id in self.buffer:
            for x, y, logits in self.buffer[task_id]:
                all_samples.append((task_id, x, y, logits))
        return all_samples
    
    def sample_batch(self, batch_size):
        """Sample a balanced batch from all tasks."""
        all_samples = self.get_all_samples()
        if len(all_samples) == 0:
            return None
        
        # Balance sampling across tasks
        tasks = list(self.buffer.keys())
        samples_per_task = max(1, batch_size // len(tasks))
        
        sampled = []
        for task_id in tasks:
            task_samples = [(task_id, x, y, logits) for x, y, logits in self.buffer[task_id]]
            if len(task_samples) > 0:
                indices = np.random.choice(len(task_samples), 
                                         min(samples_per_task, len(task_samples)),
                                         replace=False)
                sampled.extend([task_samples[i] for i in indices])
        
        # If we don't have enough samples, add random ones
        if len(sampled) < batch_size and len(all_samples) > len(sampled):
            remaining = batch_size - len(sampled)
            additional = np.random.choice(len(all_samples), min(remaining, len(all_samples)), replace=False)
            sampled.extend([all_samples[i] for i in additional])
        
        # Return list of samples (logits have different sizes per task)
        return sampled

# -------------------------------
# Standard DER++ Training
# -------------------------------
def train_derpp_standard():
    """Main training function for standard DER++ with minimal memory."""
    
    # Prepare data
    print("\n" + "="*60)
    print("Preparing Fashion-MNIST Task-Incremental Data")
    print("="*60)
    timestep_tasks, task_test_sets, task_num_classes = prepare_task_incremental_data()
    
    # Initialize model, buffer, and optimizer
    print("\n" + "="*60)
    print("Initializing Model and Components")
    print("="*60)
    model = MultitaskModel().to(device)
    buffer = DERBuffer(capacity_per_task=MEMORY_PER_TASK)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TASK)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    training_history = {
        'task_accuracies': [],
        'loss_components': defaultdict(list)
    }
    
    # -------------------------------
    # Main Training Loop - Standard DER++
    # -------------------------------
    for t, (train_ds, val_ds) in timestep_tasks.items():
        print(f"\n" + "="*60)
        print(f"TRAINING TASK {t}: {timestep_task_classes[t]}")
        print("="*60)
        
        # Add task head if new task
        if str(t) not in model.task_heads:
            model.add_task(t, task_num_classes[t])
            print(f"  Added new task head for task {t} with {task_num_classes[t]} classes")
        
        # Prepare data loaders
        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2)
        
        # Train on current task
        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            epoch_losses = {'total': [], 'current': [], 'logit': [], 'replay': []}
            
            progress_bar = tqdm(train_loader, desc=f"Task {t} Epoch {epoch+1}/{EPOCHS_PER_TASK}")
            for batch_idx, (x, y, task_ids) in enumerate(progress_bar):
                x, y = x.to(device), y.to(device)
                current_task_id = int(task_ids[0])
                
                optimizer.zero_grad()
                
                # --- 1. Current Task Loss (Standard Cross-Entropy) ---
                current_logits = model(x, current_task_id)
                loss_current = criterion(current_logits, y)
                
                # --- 2. DER++ Replay Losses ---
                loss_logit = torch.tensor(0.0, device=device)
                loss_replay = torch.tensor(0.0, device=device)
                
                replay_samples = buffer.sample_batch(BATCH_SIZE // 2)
                if replay_samples is not None:
                    # Group replay samples by task to handle different logit sizes
                    task_to_samples = {}
                    for task_id, rx, ry, r_logits in replay_samples:
                        if task_id not in task_to_samples:
                            task_to_samples[task_id] = []
                        task_to_samples[task_id].append((rx, ry, r_logits))
                    
                    # Process each task's samples separately
                    total_logit_loss = 0.0
                    total_replay_loss = 0.0
                    num_tasks_with_samples = 0
                    
                    # Save current training mode
                    training_mode = model.training
                    
                    for task_id, samples in task_to_samples.items():
                        if len(samples) == 0:
                            continue
                            
                        # Unpack samples for this task
                        task_xs, task_ys, task_logits = zip(*samples)
                        task_x = torch.stack(task_xs).to(device)
                        task_y = torch.stack(task_ys).to(device)
                        task_old_logits = torch.stack(task_logits).to(device)
                        
                        # Switch to eval mode for replay forward pass (to avoid BatchNorm issues)
                        model.eval()
                        with torch.no_grad():
                            task_new_logits = model(task_x, task_id)
                        
                        # Calculate DER++ losses for this task
                        # 2a. Logit matching loss (MSE with stored logits)
                        task_logit_loss = F.mse_loss(task_new_logits, task_old_logits)
                        # 2b. Classification loss on replay data
                        task_replay_loss = criterion(task_new_logits, task_y)
                        
                        total_logit_loss += task_logit_loss
                        total_replay_loss += task_replay_loss
                        num_tasks_with_samples += 1
                    
                    # Restore original mode
                    if training_mode:
                        model.train()
                    
                    if num_tasks_with_samples > 0:
                        loss_logit = total_logit_loss / num_tasks_with_samples
                        loss_replay = total_replay_loss / num_tasks_with_samples
                
                # --- 3. Total Loss (Standard DER++ formulation) ---
                # L_total = L_current + α * L_logit + β * L_replay
                total_loss = loss_current + ALPHA * loss_logit + BETA * loss_replay
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                
                # Track losses
                epoch_losses['total'].append(total_loss.item())
                epoch_losses['current'].append(loss_current.item())
                epoch_losses['logit'].append(loss_logit.item())
                epoch_losses['replay'].append(loss_replay.item())
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({
                        'Loss': f"{total_loss.item():.4f}",
                        'Curr': f"{loss_current.item():.4f}",
                        'Replay': f"{loss_replay.item():.4f}"
                    })
            
            # Update learning rate
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
            
            print(f"  Epoch {epoch+1}: Total Loss = {np.mean(epoch_losses['total']):.4f}, "
                  f"Val Acc = {val_acc:.3f}")
            
            # Store loss components
            for key in epoch_losses:
                training_history['loss_components'][key].append(np.mean(epoch_losses[key]))
        
        # -------------------------------
        # After Task Completion
        # -------------------------------
        
        # Select and store important samples in buffer (DER++ stores samples after each task)
        print(f"\n  Selecting important samples for task {t}...")
        buffer.add_samples(model, train_ds, t, MEMORY_PER_TASK)
        
        # Evaluate on all tasks seen so far
        print(f"\n  Evaluating on all tasks up to {t}...")
        task_accuracies = []
        with torch.no_grad():
            for task_id in range(t + 1):
                test_loader = DataLoader(task_test_sets[task_id], BATCH_SIZE, shuffle=False)
                task_correct = 0
                task_total = 0
                
                for x, y, tid in test_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x, tid[0])
                    task_correct += (preds.argmax(dim=1) == y).sum().item()
                    task_total += y.size(0)
                
                task_acc = task_correct / task_total
                task_accuracies.append(task_acc)
                print(f"    Task {task_id} accuracy: {task_acc:.3f}")
        
        training_history['task_accuracies'].append(task_accuracies)
        
        # Print summary
        avg_acc = np.mean(task_accuracies)
        print(f"\n  Average accuracy on all tasks so far: {avg_acc:.3f}")
    
    # -------------------------------
    # Final Evaluation
    # -------------------------------
    print("\n" + "="*60)
    print("FINAL EVALUATION ON ALL TASKS")
    print("="*60)
    
    model.eval()
    final_accuracies = []
    
    with torch.no_grad():
        for t, test_ds in enumerate(task_test_sets):
            test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)
            task_correct = 0
            task_total = 0
            
            for x, y, task_ids in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x, task_ids[0])
                task_correct += (preds.argmax(dim=1) == y).sum().item()
                task_total += y.size(0)
            
            task_acc = task_correct / task_total
            final_accuracies.append(task_acc)
            
            print(f"Task {t} ({timestep_task_classes[t]}): Accuracy = {task_acc:.3f}")
    
    # Summary statistics
    avg_acc = np.mean(final_accuracies)
    std_acc = np.std(final_accuracies)
    
    print("\n" + "="*60)
    print("DER++ RESULTS SUMMARY")
    print("="*60)
    print(f"Average Accuracy: {avg_acc:.3f} ± {std_acc:.3f}")
    print(f"Minimum Accuracy: {min(final_accuracies):.3f}")
    print(f"Maximum Accuracy: {max(final_accuracies):.3f}")
    
    
    # Memory usage report
    print(f"\nMemory Usage:")
    total_samples = sum(len(buffer.buffer[t]) for t in buffer.buffer)
    print(f"  Stored samples per task: {MEMORY_PER_TASK}")
    print(f"  Total stored samples: {total_samples}")
    print(f"  Buffer memory: ~{total_samples * 1.28:.1f}KB (estimated)")
    
    # Print DER++ configuration
    print(f"\nDER++ Configuration:")
    print(f"  α (logit matching weight): {ALPHA}")
    print(f"  β (replay classification weight): {BETA}")
    print(f"  Memory per task: {MEMORY_PER_TASK} samples")
    print(f"  Buffer strategy: Importance sampling (confidence + diversity)")
    
    return model, training_history, final_accuracies

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Define task classes for reference (same as your LwF code)
    timestep_task_classes = {
        0: ['Pullover', 'Dress'],
        1: ['Trouser', 'Bag'],
        2: ['Ankle boot', 'Sneaker', 'Sandal'],
        3: ['T-shirt/top', 'Coat', 'Shirt'],
    }
    
    print("="*60)
    print("STANDARD DER++ with Minimal Memory (10 examples per task)")
    print("Task-Incremental Learning on Fashion-MNIST")
    print("="*60)
    print("Algorithm: Pure DER++ (no distillation)")
    print(f"Target: Achieve ≥ 0.70 average accuracy with minimal memory")
    print("="*60)
    
    # Run training
    model, history, accuracies = train_derpp_standard()
    
    # Save model
    torch.save(model.state_dict(), "derpp_standard_model.pth")
    print(f"\nModel saved to 'derpp_standard_model.pth'")
    
    print("\nTraining completed!")
