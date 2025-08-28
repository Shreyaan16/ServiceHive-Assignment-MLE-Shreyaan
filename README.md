# 🚀 FastAPI Image Analysis Application

A complete FastAPI web application that combines **ResNet image classification** with **Gemini AI descriptions** for comprehensive image analysis.

## 🎯 Features Implemented

### 1. **Complete Image Analysis Endpoint** (`/analyze`)
- ✅ **User uploads image** via file input
- ✅ **ResNet model predicts** class and confidence interval  
- ✅ **Gemini AI generates description** incorporating model predictions
- ✅ **Returns structured JSON** with both results

### 2. **Individual Endpoints**
- 🎯 `/classify` - ResNet classification only
- 📝 `/describe` - Gemini description only  
- 🔍 `/health` - API health check
- 📚 `/docs` - Interactive API documentation

### 3. **Web Interface**
- 🖼️ **Drag & drop file upload**
- 📊 **Real-time result visualization**
- 📈 **Probability charts for all classes**
- 📝 **Formatted AI descriptions**

## 🏗️ Project Structure

```
main/
├── app.py                          # 🆕 FastAPI application
├── frontend/
│   └── index.html                  # 🆕 Web interface
├── test_api_client.py              # 🆕 API testing client
├── requirements_api.txt            # 🆕 API dependencies
├── src/                            # Your existing modules
│   ├── models/                     # ResNet & PSO models
│   └── utils/                      # Prediction & Gemini integration
└── models/                         # Trained model files
    ├── best_resnet_model.pth       # Required for API
    └── best_model_pso.pth          # Optional
```

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Set Environment Variables** (Optional)
```bash
# For Gemini AI features
set GOOGLE_API_KEY=your_api_key_here
# Get key from: https://aistudio.google.com/app/apikey
```

### 3. **Start the API Server**
```bash
python app.py
```

### 4. **Access the Application**
- 🌐 **Web Interface**: http://localhost:8000/web
- 📚 **API Documentation**: http://localhost:8000/docs  
- 🔍 **Health Check**: http://localhost:8000/health

💡 **If localhost doesn't work, try:**
- 🌐 **Web Interface**: http://127.0.0.1:8000/web
- 📚 **API Documentation**: http://127.0.0.1:8000/docs

## 📡 API Endpoints

### **POST /analyze** - Complete Analysis
Upload an image for full ResNet + Gemini analysis.

**Description:** This endpoint provides comprehensive image analysis by combining ResNet classification with Gemini AI description generation.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@your_image.jpg" \
  -F "model_type=resnet"
```

**Parameters:**
- `file` (required): Image file (JPG, JPEG, PNG, max 10MB)
- `model_type` (optional): "resnet" or "pso" (default: "resnet")

**Response:**
```json
{
  "image_filename": "mountain.jpg",
  "model_type": "resnet",
  "classification": {
    "predicted_class": "mountain",
    "confidence": 0.8945,
    "probabilities": {
      "mountain": 0.8945,
      "forest": 0.0523,
      "sea": 0.0312,
      "buildings": 0.0156,
      "glacier": 0.0042,
      "street": 0.0022
    },
    "success": true
  },
  "description": {
    "text": "This image shows a majestic mountain landscape with snow-capped peaks under a clear blue sky. Your ResNet model correctly identified this as a 'mountain' with 89.5% confidence...",
    "success": true
  },
  "status": "success",
  "processing_time_seconds": 2.451
}
```

### **POST /classify** - Classification Only
ResNet classification without Gemini description.

**Description:** Get only the ResNet model predictions with class probabilities and confidence scores.

**Request:**
```bash
curl -X POST "http://localhost:8000/classify" \
  -F "file=@your_image.jpg" \
  -F "model_type=resnet"
```

**Parameters:**
- `file` (required): Image file (JPG, JPEG, PNG, max 10MB)
- `model_type` (optional): "resnet" or "pso" (default: "resnet")

**Response:**
```json
{
  "image_filename": "forest.jpg",
  "model_type": "resnet",
  "predicted_class": "forest",
  "confidence": 0.9234,
  "probabilities": {
    "forest": 0.9234,
    "mountain": 0.0455,
    "glacier": 0.0182,
    "buildings": 0.0089,
    "sea": 0.0034,
    "street": 0.0006
  },
  "processing_time_seconds": 0.834
}
```

### **POST /describe** - Description Only  
Gemini AI description without classification.

**Description:** Get natural language description of the image using Google Gemini AI without running classification models.

**Request:**
```bash
curl -X POST "http://localhost:8000/describe" \
  -F "file=@your_image.jpg"
```

**Parameters:**
- `file` (required): Image file (JPG, JPEG, PNG, max 10MB)

**Response:**
```json
{
  "image_filename": "sunset.jpg",
  "description": "This image captures a breathtaking sunset over a tranquil ocean. The sky is painted in vibrant shades of orange, pink, and purple, with wispy clouds creating dramatic patterns across the horizon. The sun appears as a brilliant golden orb just touching the water's surface, creating a shimmering reflection that stretches toward the viewer. The composition suggests this was taken from a beach or coastal viewpoint, with the vast expanse of water dominating the lower portion of the frame.",
  "processing_time_seconds": 1.245
}
```

### **GET /web** - Web Interface
Serves the interactive web interface for image analysis.

**Description:** Access the user-friendly web interface with drag-and-drop upload, real-time results, and visual probability charts.

**Request:**
```bash
# Simply navigate to in browser:
http://localhost:8000/web
```

**Features:**
- 🖼️ **Image Preview**: See uploaded images immediately
- 📊 **Interactive Charts**: Visual probability bars for all classes
- 🎯 **Multiple Analysis Types**: Choose between complete, classify-only, or describe-only
- 📱 **Responsive Design**: Works on desktop and mobile devices

### **GET /health** - Health Check
Check API status and model availability.

**Description:** Monitor system health, model loading status, and available features.

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-28T10:30:45Z",
  "models": {
    "resnet_loaded": true,
    "pso_loaded": true,
    "gemini_available": true
  },
  "supported_classes": ["buildings", "forest", "glacier", "mountain", "sea", "street"],
  "api_version": "1.0.0",
  "features": {
    "classification": true,
    "description": true,
    "web_interface": true
  }
}
```

### **GET /** - Root Endpoint
API information and quick navigation links.

**Description:** Welcome page with API overview and navigation links to key endpoints.

**Request:**
```bash
curl -X GET "http://localhost:8000/"
```

**Response:**
```json
{
  "message": "🤖 AI Image Analysis API",
  "description": "FastAPI application combining ResNet classification with Gemini AI descriptions",
  "version": "1.0.0",
  "quick_start": {
    "web_interface": "http://localhost:8000/web",
    "api_docs": "http://localhost:8000/docs",
    "health_check": "http://localhost:8000/health"
  },
  "endpoints": {
    "analyze": "/analyze - Complete analysis (ResNet + Gemini)",
    "classify": "/classify - Classification only (ResNet)",
    "describe": "/describe - Description only (Gemini)"
  },
  "supported_formats": ["JPG", "JPEG", "PNG"],
  "max_file_size": "10MB"
}
```

### **GET /docs** - Interactive API Documentation
Auto-generated Swagger/OpenAPI documentation.

**Description:** Interactive API documentation where you can test endpoints directly in the browser.

**Features:**
- 📚 **Complete API Spec**: All endpoints with parameters and responses
- 🧪 **Try It Out**: Test endpoints directly from the browser
- 📋 **Schema Definitions**: Detailed request/response models
- 📖 **Example Requests**: Sample curl commands and responses
  "processing_time_seconds": 2.451
}
```

### **POST /classify** - Classification Only
ResNet classification without Gemini description.

**Request:**
```bash
curl -X POST "http://localhost:8000/classify" \
  -F "file=@your_image.jpg" \
  -F "model_type=resnet"
```

### **POST /describe** - Description Only  
Gemini AI description without classification.

**Request:**
```bash
curl -X POST "http://localhost:8000/describe" \
  -F "file=@your_image.jpg"
```

### **GET /health** - Health Check
Check API status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "resnet_loaded": true,
    "gemini_available": true
  },
  "supported_classes": ["buildings", "forest", "glacier", "mountain", "sea", "street"]
}
```

## 🧪 Testing

### **1. Automated Testing**
```bash
# Test all API endpoints
python test_api_client.py
```

### **2. Web Interface Testing**
1. Open `http://localhost:8000/frontend/index.html`
2. Drag & drop an image or click to upload
3. Choose analysis type:
   - **Complete Analysis**: ResNet + Gemini
   - **Classification Only**: ResNet predictions
   - **Description Only**: Gemini description

### **3. Manual API Testing**
Use the interactive documentation at `http://localhost:8000/docs`

## 🔧 Configuration

### **Model Settings** (in `src/utils/variables.py`)
```python
PSO_MODEL_PATH = "models/best_model_pso.pth"
RESNET_MODEL_PATH = "models/best_resnet_model.pth"
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
```

### **API Settings** (in `app.py`)
```python
# Server configuration
HOST = "0.0.0.0"
PORT = 8000
UPLOAD_DIR = "temp_uploads"  # Temporary file storage
MAX_FILE_SIZE = 10MB        # File size limit
```

### **Environment Variables**
- `GOOGLE_API_KEY`: Required for Gemini AI features
- `CUDA_VISIBLE_DEVICES`: Optional GPU selection

## 🎨 Web Interface Features

### **Upload Methods**
- 📁 **Click to browse** files
- 🖱️ **Drag & drop** images
- 📱 **Mobile-friendly** interface

### **Analysis Options**
- 🚀 **Complete Analysis**: Full ResNet + Gemini integration
- 🎯 **Classification Only**: Quick ResNet predictions
- 📝 **Description Only**: Pure Gemini AI descriptions

### **Results Display**
- 🎯 **Predicted class** with confidence
- 📊 **Probability bars** for all 6 classes
- 📝 **Natural language descriptions**
- ⏱️ **Processing time** metrics
- 🖼️ **Image preview** with results

## 🔍 Error Handling

### **Common Error Responses**
```json
// Invalid file type
{
  "error": "File must be an image (jpg, jpeg, png)",
  "status_code": 400
}

// Model not loaded
{
  "error": "ResNet model not loaded. Please check server logs.",
  "status_code": 503
}

// Gemini not available
{
  "error": "Gemini AI not available. Set GOOGLE_API_KEY environment variable.",
  "status_code": 503
}
```

### **Status Codes**
- `200`: Success
- `400`: Bad request (invalid file/format)
- `500`: Internal server error (processing failed)
- `503`: Service unavailable (model not loaded)

## 🚦 Production Considerations

### **Security**
```python
# Update CORS settings for production
allow_origins=["https://yourdomain.com"]  # Replace "*"

# Add authentication if needed
# Add request rate limiting
# Validate file sizes and types
```

### **Performance**
```python
# Model loading optimization
# GPU memory management
# File cleanup scheduling
# Response caching
```

### **Deployment**
```bash
# Docker deployment
# Kubernetes scaling
# Load balancer configuration
# Monitoring and logging
```

## 🎉 Success Metrics

✅ **Complete Feature Implementation**
- ✅ Image upload from user
- ✅ ResNet model classification with confidence  
- ✅ Gemini AI description generation
- ✅ Integrated response with both results

✅ **Robust API Design**
- ✅ RESTful endpoints
- ✅ Proper error handling
- ✅ File validation and cleanup
- ✅ Interactive documentation

✅ **User-Friendly Interface**
- ✅ Drag & drop upload
- ✅ Real-time results
- ✅ Visual probability charts
- ✅ Mobile responsiveness

✅ **Testing & Documentation**
- ✅ Automated test client
- ✅ Health check endpoint
- ✅ Comprehensive documentation
- ✅ Example usage code

## 💻 Example Usage in Code

### **Python Client**
```python
import requests

# Complete analysis
with open('image.jpg', 'rb') as f:
    files = {'file': ('image.jpg', f, 'image/jpeg')}
    data = {'model_type': 'resnet'}
    response = requests.post('http://localhost:8000/analyze', files=files, data=data)
    result = response.json()
    
print(f"Predicted: {result['classification']['predicted_class']}")
print(f"Confidence: {result['classification']['confidence']:.3f}")
print(f"Description: {result['description']['text']}")
```

### **JavaScript/Frontend**
```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('model_type', 'resnet');

const response = await fetch('http://localhost:8000/analyze', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log('Classification:', result.classification);
console.log('Description:', result.description);
```

Your FastAPI application is now **fully implemented** with all requested features! 🎉

- ✅ **Image upload from user**
- ✅ **ResNet model classification with confidence interval**  
- ✅ **Gemini AI description with model predictions**
- ✅ **Complete web interface for testing**
- ✅ **Comprehensive API documentation**
