import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# 1. Load Pretrained ResNet Model
model = models.resnet18(pretrained=True)

# Modify the last layer for binary classification (pneumonia vs normal)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 output classes: Normal, Pneumonia

# Print model architecture
print(model)

# 2. Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# 3. Load Chest X-ray Data
train_dataset = ImageFolder(root="data/chest_xray/train", transform=transform)
val_dataset = ImageFolder(root="data/chest_xray/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Check class names
print("Class names:", train_dataset.classes)

# 4. Move Model to GPU if Available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 6. Train the Model
num_epochs = 3  # Short training for quick testing

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 7. Evaluate the Model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
