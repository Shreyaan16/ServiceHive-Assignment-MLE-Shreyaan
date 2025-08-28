import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from utils.variables import class_names_label , IMAGE_SIZE ,device , class_names , nb_classes
from sklearn.utils import shuffle
from models.resnet import ResNet18



def load_data_resnet():
    datasets = ["data/train", "data/test"]
    output = []

    for dataset in datasets:
        images = []
        labels = []
        print(f"Loading {dataset}")

        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE) 
        
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')   
        
        output.append((images, labels))

    return output



def prepare_data_resnet(train_images, train_labels, test_images, test_labels, batch_size=32):
    # Convert from (N, H, W, C) to (N, C, H, W) for PyTorch
    train_images = np.transpose(train_images, (0, 3, 1, 2))
    test_images = np.transpose(test_images, (0, 3, 1, 2))
    
    # Normalize to [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Convert to PyTorch tensors
    train_images = torch.FloatTensor(train_images)
    train_labels = torch.LongTensor(train_labels)
    test_images = torch.FloatTensor(test_images)
    test_labels = torch.LongTensor(test_labels)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader



def train_resnet_model(model, train_dataloader, test_dataloader, epochs=50, 
                      learning_rate=0.001, weight_decay=1e-4):
    model.to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        for batch_images, batch_labels in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_labels.size(0)
            total_correct += (predicted == batch_labels).sum().item()
            total_loss += loss.item()
        
        train_accuracy = total_correct / total_samples
        train_loss = total_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_images, batch_labels in test_dataloader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                outputs = model(batch_images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_accuracy = val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'Val Acc: {val_accuracy:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'train_accuracy': train_accuracy,
                'class_names': class_names
            }, 'best_resnet_model.pth')
            print(f'New best model saved with validation accuracy: {val_accuracy:.4f}')
    
    return train_losses, train_accuracies, val_accuracies





def evaluate_resnet_model(model, test_dataloader):
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_images, batch_labels in test_dataloader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return accuracy, all_predictions, all_labels



def plot_training_history(train_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()



(train_images, train_labels), (test_images, test_labels) = load_data_resnet()
train_images, train_labels = shuffle(train_images, train_labels, random_state=42)
n_train = train_labels.shape[0]
n_test = test_labels.shape[0]
train_dataloader, test_dataloader = prepare_data_resnet(
        train_images, train_labels, test_images, test_labels, batch_size=32
    )
model = ResNet18(num_classes=nb_classes, dropout_rate=0.5)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

train_losses, train_accuracies, val_accuracies = train_resnet_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

checkpoint = torch.load('best_resnet_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_accuracy, predictions, true_labels = evaluate_resnet_model(model, test_dataloader)