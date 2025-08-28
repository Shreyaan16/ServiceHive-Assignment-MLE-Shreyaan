"""
Easy-to-use API for image classification models
Usage examples for API development
"""

# Example 1: Simple prediction
from utils import ModelManager, ImagePredictor

# Initialize
manager = ModelManager()
predictor = ImagePredictor()

# Load models
manager.load_resnet_model()
manager.load_pso_model()

# Make predictions
result = predictor.predict("image.jpg", "resnet")
print(f"Prediction: {result[0]}, Confidence: {result[1]:.4f}")

# Example 2: Compare models
comparison = predictor.compare_models("image.jpg")

# Example 3: Batch prediction
results = predictor.predict_batch(["img1.jpg", "img2.jpg"], "resnet")

# Example 4: One-liner imports
from utils.model_manager import load_resnet_model, get_model
from utils.predictor import predict_image, compare_models_on_image

# Load and predict in one go
model = load_resnet_model()
prediction = predict_image("image.jpg", "resnet")
