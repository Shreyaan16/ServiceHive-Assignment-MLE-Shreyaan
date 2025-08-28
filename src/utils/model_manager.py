"""
Model Manager - Universal model loading and management system
"""
import os
import torch
from typing import Optional, Dict, Any, Tuple

# Try relative imports first, fallback to absolute imports
try:
    from .variables import device, PSO_MODEL_PATH, RESNET_MODEL_PATH, nb_classes
    from ..models.cnn_model import CNNModel
    from ..models.resnet import ResNet18
except ImportError:
    # Fallback for when imported as top-level module
    from utils.variables import device, PSO_MODEL_PATH, RESNET_MODEL_PATH, nb_classes
    from models.cnn_model import CNNModel
    from models.resnet import ResNet18


class ModelManager:
    """
    Universal model manager for loading and managing different model types
    """
    
    def __init__(self):
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
    
    def load_pso_model(self, model_path: str = PSO_MODEL_PATH) -> torch.nn.Module:
        """
        Load PSO-optimized CNN model
        
        Args:
            model_path: Path to the PSO model file
            
        Returns:
            Loaded PSO model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PSO model file '{model_path}' not found.")
        
        try:
            # Load checkpoint with PyTorch 2.6+ compatibility
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Create model with saved parameters
            model = CNNModel(num_filters=checkpoint['num_filters'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Store model and info
            self.loaded_models['pso'] = model
            self.model_info['pso'] = {
                'type': 'PSO-CNN',
                'num_filters': checkpoint['num_filters'],
                'learning_rate': checkpoint['learning_rate'],
                'test_accuracy': checkpoint['test_accuracy'],
                'input_size': (150, 150),
                'model_path': model_path
            }
            
            print(f"✅ PSO model loaded successfully!")
            print(f"   - Filters: {checkpoint['num_filters']}")
            print(f"   - Test Accuracy: {checkpoint['test_accuracy']:.4f}")
            
            return model
            
        except Exception as e:
            raise Exception(f"Error loading PSO model: {str(e)}")
    
    def load_resnet_model(self, model_path: str = RESNET_MODEL_PATH) -> torch.nn.Module:
        """
        Load ResNet model
        
        Args:
            model_path: Path to the ResNet model file
            
        Returns:
            Loaded ResNet model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ResNet model file '{model_path}' not found.")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Create model with same architecture
            model = ResNet18(num_classes=nb_classes, dropout_rate=0.5)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Store model and info
            self.loaded_models['resnet'] = model
            self.model_info['resnet'] = {
                'type': 'ResNet-18',
                'num_classes': nb_classes,
                'dropout_rate': 0.5,
                'val_accuracy': checkpoint['val_accuracy'],
                'epoch': checkpoint['epoch'] + 1,
                'input_size': (224, 224),
                'model_path': model_path
            }
            
            print(f"✅ ResNet model loaded successfully!")
            print(f"   - Validation Accuracy: {checkpoint['val_accuracy']:.4f}")
            print(f"   - Trained Epochs: {checkpoint['epoch'] + 1}")
            
            return model
            
        except Exception as e:
            raise Exception(f"Error loading ResNet model: {str(e)}")
    
    def get_model(self, model_type: str) -> Optional[torch.nn.Module]:
        """
        Get a loaded model by type
        
        Args:
            model_type: 'pso' or 'resnet'
            
        Returns:
            Loaded model or None if not found
        """
        return self.loaded_models.get(model_type.lower())
    
    def get_model_info(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get model information
        
        Args:
            model_type: 'pso' or 'resnet'
            
        Returns:
            Model information dictionary or None if not found
        """
        return self.model_info.get(model_type.lower())
    
    def list_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all loaded models
        
        Returns:
            Dictionary with model information
        """
        return self.model_info.copy()
    
    def unload_model(self, model_type: str) -> bool:
        """
        Unload a model from memory
        
        Args:
            model_type: 'pso' or 'resnet'
            
        Returns:
            True if model was unloaded, False if not found
        """
        model_type = model_type.lower()
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
            del self.model_info[model_type]
            
            # Clear GPU cache if using CUDA
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print(f"✅ {model_type.upper()} model unloaded from memory")
            return True
        return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Attempt to load all available models
        
        Returns:
            Dictionary showing which models were successfully loaded
        """
        results = {}
        
        # Try to load PSO model
        try:
            self.load_pso_model()
            results['pso'] = True
        except Exception as e:
            print(f"❌ Failed to load PSO model: {e}")
            results['pso'] = False
        
        # Try to load ResNet model
        try:
            self.load_resnet_model()
            results['resnet'] = True
        except Exception as e:
            print(f"❌ Failed to load ResNet model: {e}")
            results['resnet'] = False
        
        return results
    
    def clear_all_models(self):
        """
        Clear all loaded models from memory
        """
        self.loaded_models.clear()
        self.model_info.clear()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print("✅ All models cleared from memory")


# Global model manager instance
model_manager = ModelManager()


# Convenience functions for easy importing
def load_pso_model(model_path: str = PSO_MODEL_PATH) -> torch.nn.Module:
    """Load PSO model using global manager"""
    return model_manager.load_pso_model(model_path)


def load_resnet_model(model_path: str = RESNET_MODEL_PATH) -> torch.nn.Module:
    """Load ResNet model using global manager"""
    return model_manager.load_resnet_model(model_path)


def get_model(model_type: str) -> Optional[torch.nn.Module]:
    """Get loaded model using global manager"""
    return model_manager.get_model(model_type)
