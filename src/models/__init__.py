# Models package
try:
    from .resnet import ResNet18, ResNet34
    from .cnn_model import CNNModel
except ImportError:
    # Fallback for direct imports
    from models.resnet import ResNet18, ResNet34
    from models.cnn_model import CNNModel

__all__ = ['ResNet18', 'ResNet34', 'CNNModel']
