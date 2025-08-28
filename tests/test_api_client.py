"""
FastAPI Test Client - Test the Image Analysis API
"""
import requests
import os
import json
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "data/pred" 

def test_api_health():
    """Test API health endpoint"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy!")
            print(f"   ResNet Model: {'✅' if data['models']['resnet_loaded'] else '❌'}")
            print(f"   Gemini AI: {'✅' if data['models']['gemini_available'] else '❌'}")
            print(f"   Classes: {data['supported_classes']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_complete_analysis(image_path: str):
    """Test complete analysis endpoint (ResNet + Gemini)"""
    print(f"\n🚀 Testing Complete Analysis: {os.path.basename(image_path)}")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {'model_type': 'resnet'}
            
            response = requests.post(f"{API_BASE_URL}/analyze", files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ Complete Analysis Successful!")
            print(f"   🎯 Predicted Class: {result['classification']['predicted_class']}")
            print(f"   📊 Confidence: {result['classification']['confidence']:.4f}")
            print(f"   ⏱️ Processing Time: {result['processing_time_seconds']}s")
            print(f"   📝 Description: {result['description']['text'][:100]}...")
            
            # Show top 3 probabilities
            probs = result['classification']['probabilities']
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print("   📈 Top 3 Predictions:")
            for class_name, prob in sorted_probs:
                print(f"      {class_name}: {prob:.4f}")
            
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def test_classification_only(image_path: str):
    """Test classification-only endpoint"""
    print(f"\n🎯 Testing Classification Only: {os.path.basename(image_path)}")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {'model_type': 'resnet'}
            
            response = requests.post(f"{API_BASE_URL}/classify", files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ Classification Successful!")
            print(f"   🎯 Predicted: {result['predicted_class']}")
            print(f"   📊 Confidence: {result['confidence']:.4f}")
            print(f"   ⏱️ Processing Time: {result['processing_time_seconds']}s")
            return True
        else:
            print(f"❌ Classification failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return False

def test_description_only(image_path: str):
    """Test description-only endpoint"""
    print(f"\n📝 Testing Description Only: {os.path.basename(image_path)}")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            
            response = requests.post(f"{API_BASE_URL}/describe", files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ Description Successful!")
            print(f"   📝 Description: {result['description']}")
            print(f"   ⏱️ Processing Time: {result['processing_time_seconds']}s")
            return True
        else:
            print(f"❌ Description failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Description error: {e}")
        return False

def find_test_image():
    """Find a test image in the test directories"""
    test_dirs = ['data/pred', 'data/test']
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            # For pred directory, look for images directly
            if test_dir == 'data/pred':
                image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    return os.path.join(test_dir, image_files[0])
            
            # For test directory, look in subdirectories
            else:
                for class_folder in os.listdir(test_dir):
                    class_path = os.path.join(test_dir, class_folder)
                    if os.path.isdir(class_path):
                        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if image_files:
                            return os.path.join(class_path, image_files[0])
    
    return None

def main():
    """Run all API tests"""
    print("=" * 80)
    print("🧪 FASTAPI IMAGE ANALYSIS - CLIENT TESTING")
    print("=" * 80)
    print("Make sure the API server is running: python app.py")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 80)
    
    # Test 1: Health check
    health_ok = test_api_health()
    if not health_ok:
        print("\n❌ API is not responding. Make sure it's running with: python app.py")
        return
    
    # Find test image
    test_image = find_test_image()
    if not test_image:
        print("\n❌ No test images found in data/pred or data/test directories")
        return
    
    print(f"\n📁 Using test image: {test_image}")
    
    # Test 2: Classification only
    classify_ok = test_classification_only(test_image)
    
    # Test 3: Description only (if Gemini is available)
    describe_ok = test_description_only(test_image)
    
    # Test 4: Complete analysis
    complete_ok = test_complete_analysis(test_image)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ API Health: {'PASS' if health_ok else 'FAIL'}")
    print(f"✅ Classification: {'PASS' if classify_ok else 'FAIL'}")
    print(f"✅ Description: {'PASS' if describe_ok else 'FAIL'}")
    print(f"✅ Complete Analysis: {'PASS' if complete_ok else 'FAIL'}")
    
    if health_ok and classify_ok:
        print("\n🎉 API is working!")
        print("You can now:")
        print("  • Upload images via http://localhost:8000/docs")
        print("  • Use the API endpoints programmatically")
        print("  • Build frontend applications that consume this API")
        
        if not describe_ok:
            print("\n💡 To enable Gemini features:")
            print("  • Set GOOGLE_API_KEY environment variable")
            print("  • Get API key from: https://aistudio.google.com/app/apikey")
    else:
        print("\n⚠️  Some tests failed. Check the API server logs.")

if __name__ == "__main__":
    main()
