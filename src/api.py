"""
Main API Interface - Single entry point for all model operations
Perfect for Flask/FastAPI integration
"""

from typing import Dict, List, Tuple, Optional, Any
import os
from utils.model_manager import ModelManager
from utils.predictor import ImagePredictor


class ImageClassificationAPI:
    """
    Main API class that provides a clean interface for image classification
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.predictor = ImagePredictor()
        self._initialized = False
    
    def initialize(self, models: List[str] = None) -> Dict[str, bool]:
        """
        Initialize the API by loading specified models
        
        Args:
            models: List of models to load ['pso', 'resnet']. If None, loads all available
            
        Returns:
            Dictionary showing which models were successfully loaded
        """
        if models is None:
            models = ['pso', 'resnet']
        
        results = {}
        
        for model_type in models:
            try:
                if model_type.lower() == 'pso':
                    self.model_manager.load_pso_model()
                    results['pso'] = True
                elif model_type.lower() == 'resnet':
                    self.model_manager.load_resnet_model()
                    results['resnet'] = True
                else:
                    print(f"⚠️  Unknown model type: {model_type}")
                    results[model_type] = False
            except Exception as e:
                print(f"❌ Failed to load {model_type}: {e}")
                results[model_type] = False
        
        self._initialized = any(results.values())
        
        if self._initialized:
            print("🚀 API initialized successfully!")
        else:
            print("❌ API initialization failed - no models loaded")
        
        return results
    
    def predict_single(self, image_path: str, model_type: str = 'resnet') -> Dict[str, Any]:
        """
        Predict a single image
        
        Args:
            image_path: Path to image file
            model_type: 'pso' or 'resnet'
            
        Returns:
            Dictionary with prediction results
        """
        if not self._initialized:
            return {"error": "API not initialized. Call initialize() first."}
        
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        predicted_class, confidence, probabilities = self.predictor.predict(image_path, model_type)
        
        if predicted_class is None:
            return {"error": f"Failed to predict image: {image_path}"}
        
        return {
            "success": True,
            "image_path": image_path,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "probabilities": {
                class_name: float(prob) 
                for class_name, prob in zip(self.predictor.class_names, probabilities)
            },
            "model_type": model_type
        }
    
    def predict_batch(self, image_paths: List[str], model_type: str = 'resnet') -> Dict[str, Any]:
        """
        Predict multiple images
        
        Args:
            image_paths: List of image file paths
            model_type: 'pso' or 'resnet'
            
        Returns:
            Dictionary with batch prediction results
        """
        if not self._initialized:
            return {"error": "API not initialized. Call initialize() first."}
        
        results = self.predictor.predict_batch(image_paths, model_type)
        
        formatted_results = []
        for result in results:
            formatted_result = {
                "image_path": result['image_path'],
                "predicted_class": result['predicted_class'],
                "confidence": float(result['confidence']),
                "probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.predictor.class_names, result['probabilities'])
                },
                "model_type": result['model_type']
            }
            formatted_results.append(formatted_result)
        
        return {
            "success": True,
            "total_images": len(image_paths),
            "successful_predictions": len(formatted_results),
            "results": formatted_results
        }
    
    def compare_models(self, image_path: str) -> Dict[str, Any]:
        """
        Compare predictions from all loaded models
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with comparison results
        """
        if not self._initialized:
            return {"error": "API not initialized. Call initialize() first."}
        
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        comparison_results = self.predictor.compare_models(image_path)
        
        if not comparison_results:
            return {"error": "No models available for comparison"}
        
        formatted_comparison = {}
        for model_type, result in comparison_results.items():
            formatted_comparison[model_type] = {
                "predicted_class": result['predicted_class'],
                "confidence": float(result['confidence']),
                "probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.predictor.class_names, result['probabilities'])
                },
                "model_info": result['model_info']
            }
        
        return {
            "success": True,
            "image_path": image_path,
            "model_comparison": formatted_comparison
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with model information
        """
        return {
            "loaded_models": self.model_manager.list_loaded_models(),
            "available_classes": self.predictor.class_names,
            "initialized": self._initialized
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint
        
        Returns:
            Dictionary with API status
        """
        loaded_models = self.model_manager.list_loaded_models()
        
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "loaded_models": list(loaded_models.keys()),
            "total_models_loaded": len(loaded_models),
            "available_classes": self.predictor.class_names,
            "device": str(self.predictor.device)
        }
    
    def unload_models(self) -> Dict[str, Any]:
        """
        Unload all models from memory
        
        Returns:
            Dictionary with operation status
        """
        self.model_manager.clear_all_models()
        self._initialized = False
        
        return {
            "success": True,
            "message": "All models unloaded from memory"
        }


# Global API instance for easy importing
api = ImageClassificationAPI()


# Convenience functions for simple usage
def quick_predict(image_path: str, model_type: str = 'resnet') -> Dict[str, Any]:
    """
    Quick prediction function that auto-initializes if needed
    
    Args:
        image_path: Path to image file
        model_type: 'pso' or 'resnet'
        
    Returns:
        Prediction result dictionary
    """
    if not api._initialized:
        print("🔄 Auto-initializing API...")
        api.initialize([model_type])
    
    return api.predict_single(image_path, model_type)


def setup_api() -> ImageClassificationAPI:
    """
    Setup and return initialized API instance
    
    Returns:
        Initialized API instance
    """
    api.initialize()
    return api
