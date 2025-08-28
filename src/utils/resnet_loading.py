# Image Prediction Functions
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
from models.resnet import ResNet18
from utils.variables import nb_classes , class_names , device

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image for prediction
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert from (H, W, C) to (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension (1, C, H, W)
        image = np.expand_dims(image, axis=0)
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image)
        
        return image_tensor
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def predict_single_image(model, image_path, class_names, device):
    """
    Predict the class of a single image
    """
    # Preprocess image
    image_tensor = load_and_preprocess_image(image_path)
    
    if image_tensor is None:
        return None, None, None
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        outputs = model(image_tensor)
        
        # Get probabilities using softmax
        probabilities = F.softmax(outputs, dim=1)
        
        # Get predicted class
        _, predicted_idx = torch.max(outputs, 1)
        predicted_idx = predicted_idx.item()
        
        # Get confidence score
        confidence = probabilities[0][predicted_idx].item()
        
        # Get predicted class name
        predicted_class = class_names[predicted_idx]
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()

def predict_multiple_images(model, image_paths, class_names, device):
    """
    Predict classes for multiple images
    """
    results = []
    
    for image_path in image_paths:
        print(f"Predicting: {os.path.basename(image_path)}")
        
        predicted_class, confidence, probabilities = predict_single_image(
            model, image_path, class_names, device
        )
        
        if predicted_class is not None:
            results.append({
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities
            })
            
            print(f"  Predicted: {predicted_class} (Confidence: {confidence:.4f})")
        else:
            print(f"  Failed to predict")
        
        print("-" * 40)
    
    return results

def visualize_predictions(results, class_names, num_images=6):
    """
    Visualize prediction results with images and probability bars
    """
    if not results:
        print("No results to visualize")
        return
    
    # Limit number of images to display
    results = results[:num_images]
    
    # Calculate grid size
    cols = 3
    rows = (len(results) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows * 2, cols, figsize=(15, rows * 6))
    
    if rows == 1:
        axes = axes.reshape(2, cols)
    
    for idx, result in enumerate(results):
        col = idx % cols
        row_img = (idx // cols) * 2
        row_prob = row_img + 1
        
        # Load and display image
        image = cv2.imread(result['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        axes[row_img, col].imshow(image)
        axes[row_img, col].set_title(
            f"Predicted: {result['predicted_class']}\n"
            f"Confidence: {result['confidence']:.3f}",
            fontsize=12
        )
        axes[row_img, col].axis('off')
        
        # Display probability distribution
        probabilities = result['probabilities']
        colors = ['red' if i == class_names.index(result['predicted_class']) else 'blue' 
                 for i in range(len(class_names))]
        
        bars = axes[row_prob, col].bar(class_names, probabilities, color=colors, alpha=0.7)
        axes[row_prob, col].set_title('Class Probabilities', fontsize=10)
        axes[row_prob, col].set_ylabel('Probability')
        axes[row_prob, col].set_ylim(0, 1)
        axes[row_prob, col].tick_params(axis='x', rotation=45)
        
        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            axes[row_prob, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Hide empty subplots
    total_subplots = rows * cols
    for idx in range(len(results), total_subplots):
        col = idx % cols
        row_img = (idx // cols) * 2
        row_prob = row_img + 1
        axes[row_img, col].axis('off')
        axes[row_prob, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_random_test_images(test_data_dir, num_images=6):
    """
    Get random images from test directory for prediction
    """
    image_paths = []
    
    # Collect all image paths
    for class_folder in os.listdir(test_data_dir):
        class_path = os.path.join(test_data_dir, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, image_file))
    
    # Randomly select images
    if len(image_paths) >= num_images:
        selected_images = random.sample(image_paths, num_images)
    else:
        selected_images = image_paths
    
    return selected_images


# Load the best trained model
try:
    print("Loading trained ResNet model...")
    checkpoint = torch.load('best_resnet_model.pth', map_location=device)
    
    # Create model with same architecture
    loaded_model = ResNet18(num_classes=nb_classes, dropout_rate=0.5)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.to(device)
    loaded_model.eval()
    
    print("✅ Model loaded successfully!")
    print(f"Model validation accuracy: {checkpoint['val_accuracy']:.4f}")
    print(f"Model was trained for epoch: {checkpoint['epoch'] + 1}")
    
except FileNotFoundError:
    print("❌ Model file 'best_resnet_model.pth' not found!")
    print("Please train the model first by running the training cells above.")
    loaded_model = None

if loaded_model is not None:
    # Method 1: Predict on random test images
    print("\n" + "="*50)
    print("METHOD 1: PREDICTING RANDOM TEST IMAGES")
    print("="*50)
    
    try:
        # Get random test images
        test_data_dir = "data/test"
        random_images = get_random_test_images(test_data_dir, num_images=6)
        
        if random_images:
            print(f"Selected {len(random_images)} random test images for prediction:")
            for img_path in random_images:
                print(f"  - {img_path}")
            
            # Make predictions
            print("\nMaking predictions...")
            prediction_results = predict_multiple_images(
                loaded_model, random_images, class_names, device
            )
            
            # Visualize results
            print("\nVisualizing prediction results...")
            visualize_predictions(prediction_results, class_names)
            
        else:
            print("❌ No test images found in 'data/test' directory")
            
    except Exception as e:
        print(f"❌ Error during random image prediction: {str(e)}")
    
    # Method 2: Predict specific images from pred folder
    print("\n" + "="*50)
    print("METHOD 2: PREDICTING IMAGES FROM PRED FOLDER")
    print("="*50)
    
    try:
        pred_data_dir = "data/pred"
        if os.path.exists(pred_data_dir):
            # Get first 6 images from pred folder
            pred_images = []
            for file in os.listdir(pred_data_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    pred_images.append(os.path.join(pred_data_dir, file))
                if len(pred_images) >= 6:  # Limit to 6 images
                    break
            
            if pred_images:
                print(f"Found {len(pred_images)} images in pred folder:")
                for img_path in pred_images:
                    print(f"  - {os.path.basename(img_path)}")
                
                # Make predictions
                print("\nMaking predictions on pred images...")
                pred_results = predict_multiple_images(
                    loaded_model, pred_images, class_names, device
                )
                
                # Visualize results
                print("\nVisualizing pred folder results...")
                visualize_predictions(pred_results, class_names)
                
            else:
                print("❌ No images found in 'data/pred' directory")
        else:
            print("❌ 'data/pred' directory not found")
            
    except Exception as e:
        print(f"❌ Error during pred folder prediction: {str(e)}")
    
    # Method 3: Single image prediction example
    print("\n" + "="*50)
    print("METHOD 3: SINGLE IMAGE PREDICTION EXAMPLE")
    print("="*50)
    
    try:
        # Try to predict a single image
        single_image_path = None
        
        # Look for any image in test directory
        for class_folder in os.listdir("data/test"):
            class_path = os.path.join("data/test", class_folder)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        single_image_path = os.path.join(class_path, image_file)
                        break
                if single_image_path:
                    break
        
        if single_image_path:
            print(f"Predicting single image: {single_image_path}")
            
            predicted_class, confidence, probabilities = predict_single_image(
                loaded_model, single_image_path, class_names, device
            )
            
            if predicted_class:
                print(f"\n🎯 Prediction Results:")
                print(f"   Image: {os.path.basename(single_image_path)}")
                print(f"   Predicted Class: {predicted_class}")
                print(f"   Confidence: {confidence:.4f}")
                print(f"   True Class: {os.path.basename(os.path.dirname(single_image_path))}")
                
                print(f"\n📊 All Class Probabilities:")
                for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                    print(f"   {class_name}: {prob:.4f}")
            else:
                print("❌ Failed to predict single image")
        else:
            print("❌ No test images found for single prediction")
            
    except Exception as e:
        print(f"❌ Error during single image prediction: {str(e)}")
