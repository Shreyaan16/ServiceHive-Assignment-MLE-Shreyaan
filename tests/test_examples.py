"""
Simple usage examples for the improved model structure
Run this file to test the API functionality
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import ImageClassificationAPI, quick_predict


def example_1_basic_usage():
    """Example 1: Basic usage with API class"""
    print("=" * 60)
    print("EXAMPLE 1: Basic API Usage")
    print("=" * 60)
    
    # Initialize API
    api = ImageClassificationAPI()
    
    # Load models (try to load both)
    results = api.initialize(['resnet', 'pso'])
    print(f"Model loading results: {results}")
    
    # Health check
    health = api.health_check()
    print(f"API Health: {health}")
    
    # Predict single image (adjust path as needed)
    image_path = "data/pred/5.jpg"  # Update this path
    if os.path.exists(image_path):
        result = api.predict_single(image_path, 'resnet')
        print(f"Prediction result: {result}")
    else:
        print(f"Image not found: {image_path}")


def example_2_quick_predict():
    """Example 2: Quick prediction function"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Quick Predict Function")
    print("=" * 60)
    
    # Quick prediction (auto-initializes)
    image_path = "data/pred/5.jpg"  # Update this path
    if os.path.exists(image_path):
        result = quick_predict(image_path, 'resnet')
        print(f"Quick prediction: {result}")
    else:
        print(f"Image not found: {image_path}")


def example_3_model_comparison():
    """Example 3: Compare multiple models"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Model Comparison")
    print("=" * 60)
    
    api = ImageClassificationAPI()
    
    # Try to load both models
    api.initialize(['resnet', 'pso'])
    
    # Compare models on same image
    image_path = "data/pred/5.jpg"  # Update this path
    if os.path.exists(image_path):
        comparison = api.compare_models(image_path)
        print(f"Model comparison: {comparison}")
    else:
        print(f"Image not found: {image_path}")


def example_4_batch_prediction():
    """Example 4: Batch prediction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Prediction")
    print("=" * 60)
    
    api = ImageClassificationAPI()
    api.initialize(['resnet'])
    
    # Batch prediction (adjust paths as needed)
    image_paths = [
        "data/pred/5.jpg",
        "data/pred/10004.jpg",
        "data/pred/101.jpg"
    ]
    
    # Filter existing paths
    existing_paths = [path for path in image_paths if os.path.exists(path)]
    
    if existing_paths:
        results = api.predict_batch(existing_paths, 'resnet')
        print(f"Batch prediction results: {results}")
    else:
        print("No valid image paths found")


def example_5_simple_imports():
    """Example 5: Using simple import functions"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Simple Import Functions")
    print("=" * 60)
    
    # Direct imports for simple usage
    from utils.model_manager import load_resnet_model, get_model
    from utils.predictor import predict_image
    
    try:
        # Load model
        model = load_resnet_model()
        print("✅ ResNet model loaded via direct import")
        
        # Make prediction
        image_path = "data/pred/5.jpg"
        if os.path.exists(image_path):
            result = predict_image(image_path, 'resnet')
            print(f"Direct prediction result: {result}")
        else:
            print(f"Image not found: {image_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    """
    Run examples to test the new structure
    """
    print("🚀 Testing Improved Model Structure")
    print("Make sure you have trained models available!")
    
    # Run examples
    try:
        example_1_basic_usage()
        example_2_quick_predict()
        example_3_model_comparison()
        example_4_batch_prediction()
        example_5_simple_imports()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure:")
        print("1. You have trained models (best_resnet_model.pth, best_model_pso.pth)")
        print("2. You have test images in data/pred/ folder")
        print("3. All required packages are installed")
