# training_utils_cifar100.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os
from torchvision.models import resnet18, wide_resnet50_2

# -------------------
# 1. Set seeds
# -------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# -------------------
# 2. CIFAR-100 transforms
# -------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# -------------------
# 3. Dataset with optional class removal
# -------------------
class FilteredCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, remove_class=None, **kwargs):
        super().__init__(*args, **kwargs)
        if remove_class is not None:
            mask = np.array(self.targets) != remove_class
            self.data = self.data[mask]
            self.targets = np.array(self.targets)[mask].tolist()

def get_dataloaders(batch_size=512, remove_class=None, dataset_dir="./../data/"):
    trainset = FilteredCIFAR100(
        root=dataset_dir, train=True, download=True,
        transform=transform_train, remove_class=remove_class
    )
    testset = torchvision.datasets.CIFAR100(
        root=dataset_dir, train=False, download=True,
        transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True
    )
    return trainloader, testloader

# -------------------
# 4. Model builders
# -------------------
def build_cifar_resnet(num_classes=100, device=None):
    """ ResNet-18 adapted for CIFAR """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if device is not None:
        model = model.to(device)
    return model

def build_wideresnet(num_classes=100, device=None):
    """ WideResNet-28-10 equivalent (using torchvision’s wide_resnet50_2 as proxy) """
    model = wide_resnet50_2(weights=None)  # WideResNet-50-2 is a strong baseline
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if device is not None:
        model = model.to(device)
    return model

# -------------------
# 5. Training / Eval (same as your CIFAR-10 version)
# -------------------
def train_model(model, trainloader, criterion, optimizer, scheduler, epochs=50, save_path=None, device="cuda"):
    model.train()
    scaler = torch.amp.GradScaler(device=device, enabled=(device.startswith("cuda")))

    for epoch in range(1, epochs+1):
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device, enabled=(device.startswith("cuda"))):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {running_loss/total:.3f} | Train Acc: {100.*correct/total:.2f}%")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def evaluate_model(model, testloader, criterion, device="cuda"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets, all_confs = [], [], []

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confs.extend(probs)

    acc = 100.*correct/total
    print(f"Test Loss: {running_loss/total:.3f} | Test Acc: {acc:.2f}%")

    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_confs)
    )

def load_model(model, path, device="cuda"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"Model loaded from {path}")
    else:
        print(f"No model found at {path}")
