import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image configuration
IMAGE_SIZE = (150, 150)  # For PSO model
IMAGE_SIZE_RESNET = (224, 224)  # For ResNet model

# Class configuration
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)

# Model paths
PSO_MODEL_PATH = 'models/best_model_pso.pth'
RESNET_MODEL_PATH = 'models/best_resnet_model.pth'

# Training configuration
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.001