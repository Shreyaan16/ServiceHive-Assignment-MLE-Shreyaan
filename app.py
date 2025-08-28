"""
FastAPI Image Analysis Application
Integrates ResNet classification with Gemini AI description
"""
import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

load_dotenv()

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom modules
try:
    from utils.model_manager import model_manager
    from utils.gemini_descriptor import analyze_image, GeminiImageDescriptor
    from utils.predictor import predict_image
    from utils.variables import class_names
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(
    title="AI Image Analysis API",
    description="ResNet Classification + Gemini AI Description",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global variables for model state
MODEL_LOADED = False
GEMINI_AVAILABLE = False
UPLOAD_DIR = "temp_uploads"

# Ensure upload directory exists
Path(UPLOAD_DIR).mkdir(exist_ok=True)

# Response models
class ClassificationResult(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    success: bool

class DescriptionResult(BaseModel):
    text: str
    success: bool

class AnalysisResponse(BaseModel):
    image_filename: str
    model_type: str
    classification: ClassificationResult
    description: DescriptionResult
    status: str
    processing_time_seconds: Optional[float] = None

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models when the app starts"""
    global MODEL_LOADED, GEMINI_AVAILABLE
    
    print("🚀 Starting AI Image Analysis API...")
    
    # Try to load ResNet model
    try:
        if os.path.exists('models/best_resnet_model.pth'):
            model_manager.load_resnet_model()
            MODEL_LOADED = True
            print("✅ ResNet model loaded successfully")
        else:
            print("❌ ResNet model not found at 'models/best_resnet_model.pth'")
    except Exception as e:
        print(f"❌ Failed to load ResNet model: {e}")
    
    # Check Gemini availability
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            # Test Gemini initialization
            descriptor = GeminiImageDescriptor()
            if descriptor.gemini_model:
                GEMINI_AVAILABLE = True
                print("✅ Gemini AI available")
            else:
                print("❌ Gemini AI initialization failed")
        else:
            print("⚠️  GOOGLE_API_KEY not found - Gemini features disabled")
    except Exception as e:
        print(f"❌ Gemini check failed: {e}")
    
    print(f"📊 API Status: ResNet={MODEL_LOADED}, Gemini={GEMINI_AVAILABLE}")

# Cleanup function for uploaded files
def cleanup_file(file_path: str):
    """Remove uploaded file after processing"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Failed to cleanup file {file_path}: {e}")

# Root endpoint
@app.get("/")
async def root():
    """Welcome endpoint with API information"""
    return {
        "message": "AI Image Analysis API",
        "description": "Upload images for ResNet classification + Gemini AI description",
        "features": {
            "resnet_classification": MODEL_LOADED,
            "gemini_description": GEMINI_AVAILABLE,
            "supported_formats": ["jpg", "jpeg", "png"],
            "max_file_size": "10MB"
        },
        "quick_start": {
            "web_interface": "/web",
            "api_docs": "/docs", 
            "health_check": "/health"
        },
        "endpoints": {
            "analyze": "/analyze - Complete analysis (ResNet + Gemini)",
            "classify": "/classify - ResNet classification only", 
            "describe": "/describe - Gemini description only",
            "health": "/health - API health check",
            "docs": "/docs - Interactive API documentation",
            "web": "/web - Web interface for easy testing"
        }
    }

# Web interface endpoint
@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Serve the web interface for easy API testing"""
    try:
        if os.path.exists("frontend/index.html"):
            return FileResponse("frontend/index.html")
        else:
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Web Interface Not Found</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 40px; text-align: center; }
                    .error { color: #e74c3c; }
                    .info { color: #3498db; margin-top: 20px; }
                </style>
            </head>
            <body>
                <h1 class="error">Web Interface Not Found</h1>
                <p>The frontend/index.html file is missing.</p>
                <div class="info">
                    <p>You can still use the API endpoints:</p>
                    <ul style="text-align: left; display: inline-block;">
                        <li><a href="/docs">Interactive API Documentation</a></li>
                        <li><a href="/health">Health Check</a></li>
                        <li>POST /analyze - Complete analysis</li>
                        <li>POST /classify - Classification only</li>
                        <li>POST /describe - Description only</li>
                    </ul>
                </div>
            </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading web interface: {str(e)}</h1>")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "resnet_loaded": MODEL_LOADED,
            "gemini_available": GEMINI_AVAILABLE
        },
        "supported_classes": class_names
    }

# Complete analysis endpoint (ResNet + Gemini)
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_type: str = "resnet"
):
    """
    Complete image analysis: ResNet classification + Gemini description
    
    - **file**: Image file (jpg, jpeg, png)
    - **model_type**: Model to use ('resnet' or 'pso')
    """
    import time
    start_time = time.time()
    
    # Validate model availability
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503, 
            detail="ResNet model not loaded. Please check server logs."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, jpeg, png)"
        )
    
    # Validate file name and extension
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File must have a valid filename"
        )
    
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['jpg', 'jpeg', 'png']:
        raise HTTPException(
            status_code=400,
            detail="Supported formats: jpg, jpeg, png"
        )
    
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Perform analysis
        if GEMINI_AVAILABLE:
            # Complete analysis with Gemini
            result = analyze_image(file_path, model_type)
            
            # Format response
            classification_data = result['classification']
            description_data = result['description']
            
            # Convert probabilities array to named dictionary
            prob_dict = {}
            if classification_data.get('probabilities') is not None:
                for i, class_name in enumerate(class_names):
                    prob_dict[class_name] = float(classification_data['probabilities'][i])
            
            response = AnalysisResponse(
                image_filename=file.filename,
                model_type=model_type,
                classification=ClassificationResult(
                    predicted_class=classification_data.get('predicted_class', 'unknown'),
                    confidence=classification_data.get('confidence', 0.0),
                    probabilities=prob_dict,
                    success=classification_data.get('success', False)
                ),
                description=DescriptionResult(
                    text=description_data.get('text', 'No description available'),
                    success=description_data.get('success', False)
                ),
                status=result['status'],
                processing_time_seconds=round(time.time() - start_time, 3)
            )
        else:
            # Classification only (no Gemini)
            predicted_class, confidence, probabilities = predict_image(file_path, model_type)
            
            if predicted_class:
                prob_dict = {}
                if probabilities is not None:
                    for i, class_name in enumerate(class_names):
                        prob_dict[class_name] = float(probabilities[i])
                
                response = AnalysisResponse(
                    image_filename=file.filename or "unknown",
                    model_type=model_type,
                    classification=ClassificationResult(
                        predicted_class=predicted_class,
                        confidence=confidence or 0.0,
                        probabilities=prob_dict,
                        success=True
                    ),
                    description=DescriptionResult(
                        text="Gemini AI not available. Set GOOGLE_API_KEY for descriptions.",
                        success=False
                    ),
                    status="partial_success",
                    processing_time_seconds=round(time.time() - start_time, 3)
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Classification failed"
                )
        
        # Schedule file cleanup
        background_tasks.add_task(cleanup_file, file_path)
        
        return response
        
    except Exception as e:
        # Cleanup on error
        background_tasks.add_task(cleanup_file, file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# Classification-only endpoint
@app.post("/classify")
async def classify_image_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_type: str = "resnet"
):
    """
    ResNet classification only
    
    - **file**: Image file (jpg, jpeg, png)  
    - **model_type**: Model to use ('resnet' or 'pso')
    """
    import time
    start_time = time.time()
    
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="ResNet model not loaded"
        )
    
    # Validate and save file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a valid filename")
    
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['jpg', 'jpeg', 'png']:
        raise HTTPException(status_code=400, detail="Supported formats: jpg, jpeg, png")
    
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Perform classification
        predicted_class, confidence, probabilities = predict_image(file_path, model_type)
        
        if predicted_class:
            # Convert probabilities to named dictionary
            prob_dict = {}
            if probabilities is not None:
                for i, class_name in enumerate(class_names):
                    prob_dict[class_name] = float(probabilities[i])
            
            result = {
                "image_filename": file.filename or "unknown",
                "model_type": model_type,
                "predicted_class": predicted_class,
                "confidence": confidence or 0.0,
                "probabilities": prob_dict,
                "processing_time_seconds": round(time.time() - start_time, 3),
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="Classification failed")
        
        background_tasks.add_task(cleanup_file, file_path)
        return result
        
    except Exception as e:
        background_tasks.add_task(cleanup_file, file_path)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Description-only endpoint
@app.post("/describe")
async def describe_image_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Gemini AI description only
    
    - **file**: Image file (jpg, jpeg, png)
    """
    import time
    start_time = time.time()
    
    if not GEMINI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Gemini AI not available. Set GOOGLE_API_KEY environment variable."
        )
    
    # Validate and save file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a valid filename")
    
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['jpg', 'jpeg', 'png']:
        raise HTTPException(status_code=400, detail="Supported formats: jpg, jpeg, png")
    
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get description
        from utils.gemini_descriptor import describe_image
        description = describe_image(file_path)
        
        if description and not description.startswith("Error"):
            result = {
                "image_filename": file.filename or "unknown",
                "description": description,
                "processing_time_seconds": round(time.time() - start_time, 3),
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Description failed: {description}")
        
        background_tasks.add_task(cleanup_file, file_path)
        return result
        
    except Exception as e:
        background_tasks.add_task(cleanup_file, file_path)
        raise HTTPException(status_code=500, detail=f"Description failed: {str(e)}")

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

if __name__ == "__main__":
    # Run the application
    print("🚀 Starting FastAPI Image Analysis Server...")
    print("🌐 Web Interface: http://localhost:8000/web")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("⚠️  Make sure GOOGLE_API_KEY is set for Gemini features")
    print()
    print("💡 If localhost doesn't work, try:")
    print("   🌐 Web Interface: http://127.0.0.1:8000/web")
    print("   📚 API Documentation: http://127.0.0.1:8000/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )