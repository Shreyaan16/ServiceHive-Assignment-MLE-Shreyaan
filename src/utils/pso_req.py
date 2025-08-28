import os
import cv2
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
from variables import class_names_label , IMAGE_SIZE , device
from models.cnn_model import CNNModel
from torch.utils.data import DataLoader, TensorDataset
import random


def load_data():
    datasets = ["data/train", "data/test"]
    output = []

    for dataset in datasets:
        images = []
        labels = []
        print(f"Loading {dataset}")

        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE) 
        
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')   
        
        output.append((images, labels))

    return output


def create_model(learning_rate=0.001, num_filters=32):
    model = CNNModel(num_filters=num_filters)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def prepare_data(train_images, train_labels, test_images, test_labels, batch_size=64):
    # Convert from (N, H, W, C) to (N, C, H, W) for PyTorch
    train_images = np.transpose(train_images, (0, 3, 1, 2))
    test_images = np.transpose(test_images, (0, 3, 1, 2))
    
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


def train_model(model, optimizer, criterion, train_dataloader, epochs=10, verbose=True):
    model.to(device)
    model.train()
    
    accuracies = []
    
    for epoch in range(epochs):
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        for batch_images, batch_labels in train_dataloader:
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
        
        epoch_accuracy = total_correct / total_samples
        epoch_loss = total_loss / len(train_dataloader)
        accuracies.append(epoch_accuracy)
        
        if verbose:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
    return accuracies


def evaluate_model(model, test_dataloader):
    model.to(device)
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_images, batch_labels in test_dataloader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_labels.size(0)
            total_correct += (predicted == batch_labels).sum().item()
    
    accuracy = total_correct / total_samples
    return accuracy



def fitness_function(params, train_dataloader):
    lr = params[0]
    filters = int(params[1])
    
    model, optimizer, criterion = create_model(learning_rate=lr, num_filters=filters)
    accuracies = train_model(model, optimizer, criterion, train_dataloader, epochs=10, verbose=False)
    
    # Clean up GPU memory
    del model, optimizer, criterion
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Return the final accuracy (last epoch)
    return accuracies[-1]


def PSO(bounds, train_dataloader, n_particles=10, max_iter=2):
    dim = len(bounds)
    particles = [np.array([random.uniform(low, high) for low, high in bounds]) for _ in range(n_particles)]
    velocities = [np.zeros(dim) for _ in range(n_particles)]
    personal_best = particles.copy()
    personal_best_scores = [0] * n_particles
    global_best = None
    global_best_score = 0

    w = 0.9  # inertia weight
    c1 = 2.0  # cognitive parameter
    c2 = 2.0  # social parameter

    for it in range(max_iter):
        print(f"PSO Iteration {it+1}/{max_iter}")
        
        for i in range(n_particles):
            print(f"  Evaluating particle {i+1}/{n_particles} - LR: {particles[i][0]:.6f}, Filters: {int(particles[i][1])}")
            score = fitness_function(particles[i], train_dataloader)
            print(f"    Accuracy: {score:.4f}")
            
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = particles[i].copy()
            if score > global_best_score:
                global_best_score = score
                global_best = particles[i].copy()

        # Update particle velocities and positions
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            cognitive = c1 * r1 * (personal_best[i] - particles[i])
            social = c2 * r2 * (global_best - particles[i])
            velocities[i] = w * velocities[i] + cognitive + social
            particles[i] += velocities[i]

            # Apply bounds constraints
            for d in range(dim):
                low, high = bounds[d]
                particles[i][d] = np.clip(particles[i][d], low, high)
        
        # Decay inertia weight
        w *= 0.95
    
    return global_best, global_best_score