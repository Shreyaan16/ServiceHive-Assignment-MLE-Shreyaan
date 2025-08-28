"""
Gemini Image Descriptor - Integrates ResNet classification with Gemini AI description
"""
import os
import google.generativeai as genai
import PIL.Image
from typing import Tuple, Optional, Dict, Any

# Try relative imports first, fallback to absolute imports
try:
    from .predictor import predict_image
    from .model_manager import model_manager
    from .variables import class_names
except ImportError:
    # Fallback for when imported as top-level module
    from utils.predictor import predict_image
    from utils.model_manager import model_manager
    from utils.variables import class_names


class GeminiImageDescriptor:
    """
    Integrates local ResNet model predictions with Gemini AI image descriptions
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini Image Descriptor
        
        Args:
            api_key: Google AI API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.gemini_model = None
        self.class_names = class_names
        
        if self.api_key:
            self._initialize_gemini()
        else:
            print("⚠️  Warning: No Google API key found. Set GOOGLE_API_KEY environment variable.")
    
    def _initialize_gemini(self):
        """Initialize Gemini AI model"""
        try:
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            print("✅ Gemini AI model initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini AI: {e}")
            self.gemini_model = None
    
    def predict_and_describe(self, image_path: str, model_type: str = 'resnet') -> Dict[str, Any]:
        """
        Complete image analysis: ResNet prediction + Gemini description
        
        Args:
            image_path: Path to image file
            model_type: 'pso' or 'resnet'
            
        Returns:
            Dictionary with prediction and description results
        """
        result = {
            'image_path': image_path,
            'model_type': model_type,
            'classification': {},
            'description': {},
            'status': 'success'
        }
        
        try:
            # Step 1: Get ResNet classification
            print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
            print("Step 1: Getting model prediction...")
            
            predicted_class, confidence, probabilities = predict_image(image_path, model_type)
            
            if predicted_class is not None:
                result['classification'] = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'success': True
                }
                print(f"   ✅ Model predicts: {predicted_class} (confidence: {confidence:.4f})")
            else:
                result['classification'] = {
                    'predicted_class': None,
                    'confidence': 0.0,
                    'probabilities': None,
                    'success': False
                }
                print("   ❌ Model prediction failed")
            
            # Step 2: Get Gemini description
            print("Step 2: Getting AI description...")
            description = self._get_integrated_description(image_path, result['classification'])
            
            if description:
                result['description'] = {
                    'text': description,
                    'success': True
                }
                print(f"   ✅ Description generated")
            else:
                result['description'] = {
                    'text': "Failed to generate description",
                    'success': False
                }
                print("   ❌ Description generation failed")
            
            return result
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            print(f"❌ Error during analysis: {e}")
            return result
    
    def _get_integrated_description(self, image_path: str, classification: Dict[str, Any]) -> Optional[str]:
        """
        Generate Gemini description that incorporates model prediction
        
        Args:
            image_path: Path to image file
            classification: Classification results from ResNet
            
        Returns:
            Generated description or None if failed
        """
        if not self.gemini_model:
            return "Gemini AI not available. Please set GOOGLE_API_KEY environment variable."
        
        try:
            # Load image for Gemini
            img = PIL.Image.open(image_path)
            
            # Create dynamic prompt based on classification results
            if classification['success'] and classification['predicted_class']:
                # Success case: Model worked
                predicted_class = classification['predicted_class']
                confidence = classification['confidence']
                confidence_percent = confidence * 100
                
                prompt = (
                    f"My custom ResNet model confidently predicts this is an image of a '{predicted_class}' "
                    f"with {confidence_percent:.1f}% confidence. Please provide a detailed descriptive sentence "
                    f"that confirms or discusses this prediction while adding visual details you observe. "
                    f"Also mention my model's classification result and confidence level in your response."
                )
            else:
                # Failure case: Model failed
                prompt = (
                    "Please provide a detailed descriptive sentence for this image. "
                    "Note that my custom ResNet classification model was unable to classify this image successfully."
                )
            
            # Generate description with Gemini
            response = self.gemini_model.generate_content([prompt, img])
            return response.text
            
        except FileNotFoundError:
            return f"Error: Image file not found at {image_path}"
        except Exception as e:
            return f"Error generating description: {str(e)}"
    
    def get_simple_description(self, image_path: str) -> Optional[str]:
        """
        Get simple one-line description without model integration
        
        Args:
            image_path: Path to image file
            
        Returns:
            Simple description or None if failed
        """
        if not self.gemini_model:
            return "Gemini AI not available"
        
        try:
            img = PIL.Image.open(image_path)
            prompt = "Give a concise, one-line description of this image."
            
            response = self.gemini_model.generate_content([prompt, img])
            return response.text
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def batch_analyze(self, image_paths: list, model_type: str = 'resnet') -> list:
        """
        Analyze multiple images with prediction + description
        
        Args:
            image_paths: List of image file paths
            model_type: 'pso' or 'resnet'
            
        Returns:
            List of analysis results
        """
        results = []
        
        print(f"🚀 Starting batch analysis of {len(image_paths)} images...")
        print("=" * 60)
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            
            result = self.predict_and_describe(image_path, model_type)
            results.append(result)
            
            # Print summary
            if result['status'] == 'success':
                classification = result['classification']
                description = result['description']
                
                if classification['success']:
                    print(f"📊 Classification: {classification['predicted_class']} ({classification['confidence']:.3f})")
                
                if description['success']:
                    print(f"📝 Description: {description['text'][:100]}...")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
            print("-" * 60)
        
        return results
    
    def display_detailed_results(self, result: Dict[str, Any]):
        """
        Display detailed analysis results in a formatted way
        
        Args:
            result: Result dictionary from predict_and_describe
        """
        print("\n" + "=" * 80)
        print("🎯 COMPLETE IMAGE ANALYSIS RESULTS")
        print("=" * 80)
        
        print(f"📁 Image: {os.path.basename(result['image_path'])}")
        print(f"🤖 Model: {result['model_type'].upper()}")
        print(f"📊 Status: {result['status'].upper()}")
        
        # Classification results
        print("\n📈 CLASSIFICATION RESULTS:")
        classification = result['classification']
        if classification.get('success'):
            print(f"   🎯 Predicted Class: {classification['predicted_class']}")
            print(f"   📊 Confidence: {classification['confidence']:.4f} ({classification['confidence']*100:.1f}%)")
            
            if classification['probabilities'] is not None:
                print("   📋 All Class Probabilities:")
                for class_name, prob in zip(self.class_names, classification['probabilities']):
                    indicator = "👉" if class_name == classification['predicted_class'] else "  "
                    print(f"      {indicator} {class_name}: {prob:.4f}")
        else:
            print("   ❌ Classification failed")
        
        # Description results
        print("\n📝 AI DESCRIPTION:")
        description = result['description']
        if description.get('success'):
            print(f"   {description['text']}")
        else:
            print(f"   ❌ {description['text']}")
        
        print("=" * 80)


# Global descriptor instance
gemini_descriptor = GeminiImageDescriptor()


# Convenience functions for easy importing
def analyze_image(image_path: str, model_type: str = 'resnet') -> Dict[str, Any]:
    """Analyze single image with ResNet + Gemini using global descriptor"""
    return gemini_descriptor.predict_and_describe(image_path, model_type)


def describe_image(image_path: str) -> Optional[str]:
    """Get simple Gemini description using global descriptor"""
    return gemini_descriptor.get_simple_description(image_path)


def analyze_images(image_paths: list, model_type: str = 'resnet') -> list:
    """Analyze multiple images using global descriptor"""
    return gemini_descriptor.batch_analyze(image_paths, model_type)


def display_analysis(result: Dict[str, Any]):
    """Display analysis results using global descriptor"""
    return gemini_descriptor.display_detailed_results(result)
