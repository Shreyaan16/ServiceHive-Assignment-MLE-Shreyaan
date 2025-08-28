import os 
import torch
from models.cnn_model import CNNModel
from variables import device , IMAGE_SIZE , class_names
import cv2
import numpy as np
import torch.nn.functional as F


def load_trained_model(model_path):
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Make sure you've trained and saved the model first.")
    
    try:
        # Fix for PyTorch 2.6+ - set weights_only=False to allow loading numpy objects
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"Loading model from: {model_path}")
        
        # Create model with same parameters used during training
        model = CNNModel(num_filters=checkpoint['num_filters'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        print(f"Model loaded successfully!")
        print(f"- Filters: {checkpoint['num_filters']}")
        print(f"- Learning Rate: {checkpoint['learning_rate']}")
        print(f"- Test Accuracy: {checkpoint['test_accuracy']:.4f}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image = cv2.resize(image, IMAGE_SIZE)
    
    # Normalize pixel values
    image = image.astype('float32') / 255.0
    
    # Convert from (H, W, C) to (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    image = np.expand_dims(image, axis=0)
    
    # Convert to torch tensor
    image_tensor = torch.FloatTensor(image).to(device)
    
    return image_tensor



def predict_single_image(model, image_path):
    """Predict class for a single image"""
    try:
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        predicted_class = class_names[predicted_class_idx]
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    except Exception as e:
        print(f"Error predicting image {image_path}: {str(e)}")
        return None, None, None