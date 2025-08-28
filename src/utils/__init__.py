# Utils package
try:
    from .model_manager import ModelManager, model_manager
    from .predictor import ImagePredictor, predictor
    from .gemini_descriptor import GeminiImageDescriptor, gemini_descriptor
    from .variables import *
except ImportError:
    # Fallback for direct imports
    from utils.model_manager import ModelManager, model_manager
    from utils.predictor import ImagePredictor, predictor
    from utils.gemini_descriptor import GeminiImageDescriptor, gemini_descriptor
    from utils.variables import *

__all__ = [
    'ModelManager', 'ImagePredictor', 'GeminiImageDescriptor',
    'model_manager', 'predictor', 'gemini_descriptor',
    'device', 'class_names', 'IMAGE_SIZE', 'IMAGE_SIZE_RESNET'
]
