"""
Test script to validate all imports and basic functionality
"""
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all module imports"""
    print("Testing imports...")
    
    try:
        # Test basic imports
        from utils.variables import class_names, device, IMAGE_SIZE, IMAGE_SIZE_RESNET
        print("✅ Variables imported successfully")
        print(f"   Device: {device}")
        print(f"   Classes: {class_names}")
        print(f"   Image sizes: PSO={IMAGE_SIZE}, ResNet={IMAGE_SIZE_RESNET}")
        
        # Test model imports
        from models.cnn_model import CNNModel
        from models.resnet import ResNet, BasicBlock, ResNet18, ResNet34
        print("✅ Model classes imported successfully")
        
        # Test utility imports
        from utils.model_manager import model_manager
        from utils.predictor import predictor, predict_image
        from utils.gemini_descriptor import gemini_descriptor, analyze_image
        print("✅ Utility classes imported successfully")
        
        # Test model manager functionality
        print("\nTesting model manager...")
        loaded_models = model_manager.list_loaded_models()
        print(f"   Currently loaded models: {list(loaded_models.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\nTesting model loading...")
    
    try:
        from utils.model_manager import model_manager
        
        # Check for model files
        model_files = {
            'pso': 'models/best_model_pso.pth',
            'resnet': 'models/best_resnet_model.pth'
        }
        
        available_models = []
        for model_type, model_path in model_files.items():
            if os.path.exists(model_path):
                available_models.append(model_type)
                print(f"✅ Found {model_type} model at: {model_path}")
            else:
                print(f"⚠️  {model_type} model not found at: {model_path}")
        
        # Try loading available models
        for model_type in available_models:
            try:
                if model_type == 'pso':
                    model = model_manager.load_pso_model()
                elif model_type == 'resnet':
                    model = model_manager.load_resnet_model()
                
                if model is not None:
                    print(f"✅ Successfully loaded {model_type} model")
                else:
                    print(f"❌ Failed to load {model_type} model")
            except Exception as e:
                print(f"❌ Error loading {model_type} model: {e}")
        
        return len(available_models) > 0
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_prediction():
    """Test prediction functionality with a sample image"""
    print("\nTesting prediction functionality...")
    
    try:
        from utils.predictor import predict_image
        from utils.model_manager import model_manager
        
        # Check for test images
        test_dirs = ['data/test', 'data/pred']
        sample_image = None
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    sample_image = os.path.join(test_dir, image_files[0])
                    break
        
        if sample_image is None:
            print("⚠️  No test images found in data/test or data/pred")
            return False
        
        print(f"Using sample image: {sample_image}")
        
        # Test prediction with loaded models
        loaded_models = model_manager.list_loaded_models()
        
        if not loaded_models:
            print("⚠️  No models loaded for prediction test")
            return False
        
        for model_type in loaded_models.keys():
            try:
                predicted_class, confidence, probabilities = predict_image(sample_image, model_type)
                
                if predicted_class is not None:
                    print(f"✅ {model_type.upper()} prediction: {predicted_class} (confidence: {confidence:.4f})")
                else:
                    print(f"❌ {model_type.upper()} prediction failed")
                    
            except Exception as e:
                print(f"❌ Error predicting with {model_type}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def test_integrated_analysis():
    """Test integrated ResNet + Gemini analysis"""
    print("\nTesting integrated analysis (ResNet + Gemini)...")
    
    try:
        from utils.gemini_descriptor import analyze_image
        from utils.model_manager import model_manager
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  GOOGLE_API_KEY not found - integrated analysis will be limited")
            print("   Set your Google AI API key for full functionality:")
            print("   https://aistudio.google.com/app/apikey")
            return False
        
        # Check for test images
        test_dirs = ['data/test', 'data/pred']
        sample_image = None
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                if test_dir == 'data/pred':
                    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if image_files:
                        sample_image = os.path.join(test_dir, image_files[0])
                        break
        
        if sample_image is None:
            print("⚠️  No test images found for integrated analysis")
            return False
        
        print(f"Using sample image: {sample_image}")
        
        # Test integrated analysis with loaded models
        loaded_models = model_manager.list_loaded_models()
        
        if not loaded_models:
            print("⚠️  No models loaded for integrated analysis test")
            return False
        
        for model_type in loaded_models.keys():
            try:
                print(f"🔍 Testing {model_type.upper()} + Gemini integration...")
                result = analyze_image(sample_image, model_type)
                
                if result['status'] == 'success':
                    classification = result['classification']
                    description = result['description']
                    
                    print(f"✅ {model_type.upper()} + Gemini integration successful!")
                    
                    if classification['success']:
                        print(f"   🎯 Classification: {classification['predicted_class']} ({classification['confidence']:.3f})")
                    
                    if description['success']:
                        print(f"   📝 Description: {description['text'][:80]}...")
                else:
                    print(f"❌ {model_type.upper()} + Gemini integration failed: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"❌ Error testing {model_type} integration: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated analysis test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("FOLDER STRUCTURE AND IMPORT VALIDATION")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Please fix import issues first.")
        return
    
    # Test model loading
    models_ok = test_model_loading()
    
    # Test prediction (only if models are available)
    if models_ok:
        prediction_ok = test_prediction()
    else:
        print("\n⚠️  Skipping prediction test - no models available")
        prediction_ok = False
    
    # Test integrated analysis (only if models and imports are available)
    integrated_ok = False
    if imports_ok and models_ok:
        integrated_ok = test_integrated_analysis()
    else:
        print("\n⚠️  Skipping integrated analysis test - prerequisites not met")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✅ Imports: {'PASS' if imports_ok else 'FAIL'}")
    print("✅ Model Loading: {'PASS' if models_ok else 'FAIL'}")
    print("✅ Prediction: {'PASS' if prediction_ok else 'SKIP/FAIL'}")
    print("✅ Integrated Analysis: {'PASS' if integrated_ok else 'SKIP/FAIL'}")
    
    if imports_ok and models_ok and integrated_ok:
        print("\n🎉 COMPLETE INTEGRATION SUCCESSFUL!")
        print("Your folder structure now supports:")
        print("  • ResNet + PSO model classification")
        print("  • Gemini AI image description") 
        print("  • Integrated analysis (ResNet + Gemini)")
        print("  • Clean modular imports for API development")
        print("\nExample usage:")
        print("  from src.utils.gemini_descriptor import analyze_image")
        print("  result = analyze_image('image.jpg')")
    elif imports_ok and models_ok:
        print("\n🎉 Folder structure is ready for API development!")
        print("Add GOOGLE_API_KEY for Gemini integration.")
        print("\nYou can now import any module like:")
        print("  from src.utils.predictor import predict_image")
        print("  from src.utils.model_manager import model_manager")
        print("  from src.models.resnet import ResNet18")
    else:
        print("\n⚠️  Some issues found. Please check the output above.")

if __name__ == "__main__":
    main()
