"""
Universal Image Predictor - Clean version for easy importing
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

# Try relative imports first, fallback to absolute imports
try:
    from .variables import class_names, device, IMAGE_SIZE, IMAGE_SIZE_RESNET
    from .model_manager import model_manager
except ImportError:
    # Fallback for when imported as top-level module
    from utils.variables import class_names, device, IMAGE_SIZE, IMAGE_SIZE_RESNET
    from utils.model_manager import model_manager


class ImagePredictor:
    """
    Universal image predictor that works with any loaded model
    """
    
    def __init__(self):
        self.class_names = class_names
        self.device = device
    
    def preprocess_image(self, image_path: str, model_type: str = 'resnet') -> Optional[torch.Tensor]:
        """
        Preprocess image based on model type
        
        Args:
            image_path: Path to image file
            model_type: 'pso' or 'resnet'
            
        Returns:
            Preprocessed image tensor or None if error
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize based on model type
            if model_type.lower() == 'pso':
                target_size = IMAGE_SIZE
            else:  # resnet
                target_size = IMAGE_SIZE_RESNET
            
            image = cv2.resize(image, target_size)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert from (H, W, C) to (C, H, W)
            image = np.transpose(image, (2, 0, 1))
            
            # Add batch dimension (1, C, H, W)
            image = np.expand_dims(image, axis=0)
            
            # Convert to tensor
            image_tensor = torch.FloatTensor(image).to(self.device)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def predict(self, image_path: str, model_type: str = 'resnet') -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
        """
        Predict image class using specified model
        
        Args:
            image_path: Path to image file
            model_type: 'pso' or 'resnet'
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        try:
            # Get model
            model = model_manager.get_model(model_type)
            if model is None:
                raise ValueError(f"Model '{model_type}' not loaded. Please load it first.")
            
            # Preprocess image
            image_tensor = self.preprocess_image(image_path, model_type)
            if image_tensor is None:
                return None, None, None
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_idx = int(torch.argmax(probabilities, dim=1).item())
                confidence = float(probabilities[0][predicted_idx].item())
            
            predicted_class = self.class_names[predicted_idx]
            
            return predicted_class, confidence, probabilities[0].cpu().numpy()
            
        except Exception as e:
            print(f"Error predicting image {image_path}: {str(e)}")
            return None, None, None
    
    def predict_batch(self, image_paths: List[str], model_type: str = 'resnet') -> List[Dict[str, Any]]:
        """
        Predict multiple images
        
        Args:
            image_paths: List of image file paths
            model_type: 'pso' or 'resnet'
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            print(f"Predicting: {os.path.basename(image_path)}")
            
            predicted_class, confidence, probabilities = self.predict(image_path, model_type)
            
            if predicted_class is not None:
                result = {
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'model_type': model_type
                }
                results.append(result)
                print(f"  ✅ {predicted_class} (Confidence: {confidence:.4f})")
            else:
                print(f"  ❌ Failed to predict")
            
            print("-" * 40)
        
        return results
    
    def compare_models(self, image_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare predictions from all loaded models
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with results from each model
        """
        results = {}
        loaded_models = model_manager.list_loaded_models()
        
        if not loaded_models:
            print("❌ No models loaded. Please load models first.")
            return results
        
        print(f"Comparing models on image: {os.path.basename(image_path)}")
        print("=" * 60)
        
        for model_type in loaded_models.keys():
            predicted_class, confidence, probabilities = self.predict(image_path, model_type)
            
            if predicted_class is not None:
                results[model_type] = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'model_info': loaded_models[model_type]
                }
                
                print(f"{model_type.upper()} Model:")
                print(f"  Predicted: {predicted_class}")
                print(f"  Confidence: {confidence:.4f}")
                print(f"  Model Type: {loaded_models[model_type]['type']}")
                print("-" * 40)
            else:
                print(f"{model_type.upper()} Model: ❌ Failed to predict")
                print("-" * 40)
        
        return results


# Global predictor instance
predictor = ImagePredictor()


# Convenience functions for easy importing
def predict_image(image_path: str, model_type: str = 'resnet') -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
    """Predict single image using global predictor"""
    return predictor.predict(image_path, model_type)


def predict_images(image_paths: List[str], model_type: str = 'resnet') -> List[Dict[str, Any]]:
    """Predict multiple images using global predictor"""
    return predictor.predict_batch(image_paths, model_type)


def compare_models_on_image(image_path: str) -> Dict[str, Dict[str, Any]]:
    """Compare all loaded models on single image"""
    return predictor.compare_models(image_path)
