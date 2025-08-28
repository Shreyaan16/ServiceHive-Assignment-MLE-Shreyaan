"""
Complete Image Analysis Test - ResNet Classification + Gemini Description
"""
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic imports and model loading"""
    print("🔧 Testing basic functionality...")
    
    try:
        # Test imports
        from utils.model_manager import model_manager
        from utils.predictor import predict_image
        from utils.gemini_descriptor import analyze_image, describe_image, display_analysis
        from utils.variables import class_names
        
        print("✅ All modules imported successfully")
        
        # Check for models
        model_files = {
            'resnet': 'models/best_resnet_model.pth',
            'pso': 'models/best_model_pso.pth'
        }
        
        available_models = []
        for model_type, model_path in model_files.items():
            if os.path.exists(model_path):
                available_models.append(model_type)
                print(f"✅ Found {model_type} model")
        
        # Load at least one model
        if available_models:
            model_type = available_models[0]  # Use first available model
            if model_type == 'resnet':
                model_manager.load_resnet_model()
            else:
                model_manager.load_pso_model()
            print(f"✅ Loaded {model_type} model successfully")
            return model_type
        else:
            print("❌ No models found")
            return None
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return None

def test_classification_only(model_type: str):
    """Test ResNet classification without Gemini"""
    print("\n🎯 Testing ResNet classification...")
    
    try:
        from utils.predictor import predict_image
        
        # Find test image
        test_dirs = ['data/pred', 'data/test']
        sample_image = None
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                # Look for images in subdirectories (test) or directly (pred)
                if test_dir == 'data/test':
                    for class_folder in os.listdir(test_dir):
                        class_path = os.path.join(test_dir, class_folder)
                        if os.path.isdir(class_path):
                            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            if image_files:
                                sample_image = os.path.join(class_path, image_files[0])
                                break
                else:  # data/pred
                    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if image_files:
                        sample_image = os.path.join(test_dir, image_files[0])
                
                if sample_image:
                    break
        
        if not sample_image:
            print("❌ No test images found")
            return False
        
        print(f"📁 Testing with: {sample_image}")
        
        # Test prediction
        predicted_class, confidence, probabilities = predict_image(sample_image, model_type)
        
        if predicted_class and confidence is not None:
            print(f"✅ Classification successful!")
            print(f"   🎯 Predicted: {predicted_class}")
            print(f"   📊 Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
            
            from utils.variables import class_names
            if probabilities is not None:
                print(f"   📋 All probabilities:")
                for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                    indicator = "👉" if class_name == predicted_class else "  "
                    print(f"      {indicator} {class_name}: {prob:.4f}")
            
            return True
        else:
            print("❌ Classification failed")
            return False
            
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False

def test_gemini_description():
    """Test Gemini description without classification"""
    print("\n🤖 Testing Gemini AI description...")
    
    try:
        from utils.gemini_descriptor import describe_image
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables")
            print("   Set your Google AI API key to test Gemini functionality")
            print("   You can get one from: https://aistudio.google.com/app/apikey")
            return False
        
        # Find test image
        test_dirs = ['data/pred', 'data/test']
        sample_image = None
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                if test_dir == 'data/pred':
                    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if image_files:
                        sample_image = os.path.join(test_dir, image_files[0])
                        break
        
        if not sample_image:
            print("❌ No test images found")
            return False
        
        print(f"📁 Testing with: {sample_image}")
        
        # Test description
        description = describe_image(sample_image)
        
        if description and not description.startswith("Error"):
            print(f"✅ Gemini description successful!")
            print(f"   📝 Description: {description}")
            return True
        else:
            print(f"❌ Gemini description failed: {description}")
            return False
            
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

def test_integrated_analysis(model_type: str):
    """Test complete integrated analysis: ResNet + Gemini"""
    print("\n🚀 Testing integrated analysis (ResNet + Gemini)...")
    
    try:
        from utils.gemini_descriptor import analyze_image, display_analysis
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  Skipping integrated test - no GOOGLE_API_KEY found")
            return False
        
        # Find test image
        test_dirs = ['data/pred', 'data/test']
        sample_image = None
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                if test_dir == 'data/pred':
                    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if image_files:
                        sample_image = os.path.join(test_dir, image_files[0])
                        break
        
        if not sample_image:
            print("❌ No test images found")
            return False
        
        print(f"📁 Analyzing: {sample_image}")
        print("=" * 60)
        
        # Perform integrated analysis
        result = analyze_image(sample_image, model_type)
        
        if result['status'] == 'success':
            print("✅ Integrated analysis successful!")
            
            # Display detailed results
            display_analysis(result)
            
            return True
        else:
            print(f"❌ Integrated analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Integrated analysis test failed: {e}")
        return False

def test_batch_analysis(model_type: str):
    """Test batch analysis on multiple images"""
    print("\n📚 Testing batch analysis...")
    
    try:
        from utils.gemini_descriptor import analyze_images
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  Skipping batch test - no GOOGLE_API_KEY found")
            return False
        
        # Get multiple test images
        test_images = []
        test_dir = 'data/pred'
        
        if os.path.exists(test_dir):
            image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            test_images = [os.path.join(test_dir, f) for f in image_files[:3]]  # Limit to 3 images
        
        if not test_images:
            print("❌ No test images found for batch analysis")
            return False
        
        print(f"📁 Analyzing {len(test_images)} images...")
        
        # Perform batch analysis
        results = analyze_images(test_images, model_type)
        
        if results:
            print(f"✅ Batch analysis completed!")
            print(f"   📊 Processed: {len(results)} images")
            
            # Summary
            successful = sum(1 for r in results if r['status'] == 'success')
            print(f"   ✅ Successful: {successful}/{len(results)}")
            
            return True
        else:
            print("❌ Batch analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Batch analysis test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 80)
    print("🧪 COMPLETE IMAGE ANALYSIS TESTING")
    print("ResNet Classification + Gemini AI Description")
    print("=" * 80)
    
    # Test 1: Basic functionality
    model_type = test_basic_functionality()
    if not model_type:
        print("\n❌ Basic tests failed. Cannot proceed.")
        return
    
    # Test 2: Classification only
    classification_ok = test_classification_only(model_type)
    
    # Test 3: Gemini description only
    gemini_ok = test_gemini_description()
    
    # Test 4: Integrated analysis (if both work)
    integrated_ok = False
    if classification_ok and gemini_ok:
        integrated_ok = test_integrated_analysis(model_type)
    else:
        print("\n⚠️  Skipping integrated test - prerequisites not met")
    
    # Test 5: Batch analysis (if integrated works)
    batch_ok = False
    if integrated_ok:
        batch_ok = test_batch_analysis(model_type)
    else:
        print("\n⚠️  Skipping batch test - integrated analysis not working")
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Basic Functionality: {'PASS' if model_type else 'FAIL'}")
    print(f"✅ ResNet Classification: {'PASS' if classification_ok else 'FAIL'}")
    print(f"✅ Gemini Description: {'PASS' if gemini_ok else 'FAIL'}")
    print(f"✅ Integrated Analysis: {'PASS' if integrated_ok else 'FAIL'}")
    print(f"✅ Batch Analysis: {'PASS' if batch_ok else 'FAIL'}")
    
    if integrated_ok:
        print("\n🎉 INTEGRATION SUCCESSFUL!")
        print("Your folder structure now supports:")
        print("  • ResNet image classification")
        print("  • Gemini AI image description")
        print("  • Combined analysis with both models")
        print("  • Batch processing capabilities")
        print("\nUsage examples:")
        print("  from src.utils.gemini_descriptor import analyze_image")
        print("  result = analyze_image('path/to/image.jpg')")
        print("  display_analysis(result)")
    elif classification_ok:
        print("\n✅ CLASSIFICATION READY!")
        print("ResNet classification is working. Add GOOGLE_API_KEY for full integration.")
    else:
        print("\n⚠️  SETUP NEEDED")
        print("Please ensure models are trained and available.")

if __name__ == "__main__":
    main()
