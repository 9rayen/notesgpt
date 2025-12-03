# 🚀 NotesGPT OCR Feature - Installation & Setup Guide

## Prerequisites

Before installing the OCR feature, ensure you have:
- ✅ Python 3.10 or higher
- ✅ pip package manager
- ✅ At least 4GB RAM
- ✅ ~500MB free disk space (for OCR models)
- ✅ Virtual environment activated

## 📦 Installation Steps

### Step 1: Navigate to Project Directory

```powershell
cd c:\Users\USER\Desktop\GENAI_Project\notesgpt
```

### Step 2: Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install/Update Dependencies

```powershell
pip install -r requirements.txt
```

This will install all new OCR dependencies:
- `easyocr==1.7.1` - AI-powered OCR engine
- `opencv-python==4.8.1.78` - Image preprocessing
- Plus existing dependencies (Pillow, pdf2image, etc.)

### Step 4: Verify Installation

Run this command to check if OCR modules are installed:

```powershell
python -c "import easyocr; import cv2; print('OCR dependencies installed successfully!')"
```

Expected output:
```
OCR dependencies installed successfully!
```

## 🎯 First Run

### Step 1: Start the Application

```powershell
python -m app.main
```

Or:

```powershell
uvicorn app.main:app --reload
```

### Step 2: Open in Browser

Navigate to: http://127.0.0.1:8000

### Step 3: Test OCR Feature

1. Scroll to the **"🔍 OCR - Extract Text from Images"** section
2. Click **"📷 Choose image..."**
3. Select a test image (any image with text)
4. Click **"🔍 Extract Text (OCR)"**

### Step 4: First-Time Model Download

**Important**: On the first OCR operation, EasyOCR will download language models:

- **English model**: ~50 MB
- **French model**: ~50 MB  
- **Arabic model**: ~50 MB
- **Total**: ~150 MB

This happens automatically and takes 1-3 minutes depending on your internet speed.

**Progress indicators**:
```
Downloading detection model, please wait...
Downloading recognition model, please wait...
```

After downloading, models are cached and future OCR operations are much faster (10-30 seconds).

## 🔍 Verify OCR is Working

### Quick Test

1. Create a simple test image with text (or use a screenshot)
2. Upload to OCR section
3. Wait for processing
4. Check if text is extracted correctly
5. Review confidence scores (should be >70% for good images)

### Test API Endpoints

You can test the OCR API directly:

```powershell
# Get OCR info
Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/ocr/info" -Method GET | ConvertFrom-Json
```

Expected response:
```json
{
  "status": "success",
  "ocr": {
    "available": true,
    "engine": "EasyOCR",
    "preprocessing": true,
    "pdf_support": true,
    "default_languages": ["en", "fr", "ar"],
    "supported_languages": ["en", "fr", "ar", "es", "de", "it", "pt", "ru", "zh", "ja", "ko"],
    "supported_formats": [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"]
  }
}
```

## ⚙️ Configuration

### Default Configuration

The OCR feature is pre-configured with optimal settings:

**Languages**: English, French, Arabic (DEFAULT_LANGUAGES in `app/ocr.py`)
**GPU**: Disabled by default (uses CPU)
**Preprocessing**: Enabled for better accuracy
**Formats**: PNG, JPG, JPEG, TIFF, BMP, GIF, WebP

### Customizing Languages

To change supported languages, edit `app/ocr.py`:

```python
# Line 40-41
DEFAULT_LANGUAGES = ['en', 'fr', 'ar']  # Modify this list
```

Example for Spanish and Italian:
```python
DEFAULT_LANGUAGES = ['es', 'it']
```

**Note**: Changing languages requires downloading new models on next OCR operation.

### Enabling GPU Acceleration (Optional)

If you have an NVIDIA GPU with CUDA:

1. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
2. Install PyTorch with CUDA support:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Edit `app/ocr.py` line 65:
   ```python
   _ocr_reader = easyocr.Reader(languages, gpu=True)  # Change to True
   ```
4. Restart the application

**GPU Benefits**:
- 5-10x faster processing
- Can handle larger images
- Better for batch processing

## 🗂️ File Structure

After implementation, your project structure includes:

```
notesgpt/
├── app/
│   ├── __init__.py
│   ├── main.py           # Added OCR endpoints
│   ├── rag.py            # Updated with EasyOCR integration
│   └── ocr.py            # NEW: OCR module
├── templates/
│   └── index.html        # Updated with OCR UI
├── static/
│   ├── app.js            # Updated with OCR handlers
│   └── styles.css        # Updated with OCR styles
├── uploads/              # Stores uploaded files
├── chroma_db/            # Vector database
├── requirements.txt      # Updated with OCR dependencies
├── README.md             # Updated documentation
├── OCR_GUIDE.md          # NEW: Comprehensive OCR guide
└── INSTALLATION.md       # NEW: This file
```

## 🐛 Troubleshooting Installation

### Issue 1: pip install fails

**Error**: `Could not find a version that satisfies the requirement`

**Solution**:
```powershell
# Update pip
python -m pip install --upgrade pip

# Try again
pip install -r requirements.txt
```

### Issue 2: torch/pytorch issues

**Error**: Issues installing EasyOCR dependencies

**Solution**:
```powershell
# Install PyTorch first
pip install torch torchvision

# Then install EasyOCR
pip install easyocr
```

### Issue 3: opencv-python conflicts

**Error**: Conflicts with other opencv packages

**Solution**:
```powershell
# Uninstall all opencv versions
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y

# Install clean version
pip install opencv-python==4.8.1.78
```

### Issue 4: Memory errors during installation

**Error**: Out of memory during pip install

**Solution**:
```powershell
# Install one by one
pip install easyocr
pip install opencv-python
pip install pdf2image
```

### Issue 5: Model download fails

**Error**: Cannot download EasyOCR models

**Solutions**:
- Check internet connection
- Disable VPN/proxy if using
- Try again (downloads resume automatically)
- Manually download models (see EasyOCR docs)

### Issue 6: Import errors after installation

**Error**: `ModuleNotFoundError: No module named 'easyocr'`

**Solution**:
```powershell
# Verify virtual environment is activated
.\venv\Scripts\Activate.ps1

# Check pip list
pip list | Select-String easyocr

# Reinstall if missing
pip install easyocr
```

## 📊 System Requirements Check

Run this script to check if your system meets requirements:

```powershell
python -c "
import sys
import platform
import psutil

print(f'Python Version: {sys.version}')
print(f'Platform: {platform.system()} {platform.release()}')
print(f'RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')
print(f'Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')
print(f'CPU Cores: {psutil.cpu_count()}')

# Check if requirements met
if sys.version_info >= (3, 10):
    print('✅ Python version OK')
else:
    print('❌ Python version too old (need 3.10+)')

if psutil.virtual_memory().total >= 4 * (1024**3):
    print('✅ RAM sufficient')
else:
    print('⚠️ RAM may be insufficient (recommend 4GB+)')
"
```

## 🔄 Updating Dependencies

To update OCR packages to latest versions:

```powershell
# Update all packages
pip install --upgrade easyocr opencv-python pdf2image

# Or update entire requirements
pip install --upgrade -r requirements.txt
```

## 🧪 Testing OCR After Installation

### Test 1: Import Test

```powershell
python -c "
from app.ocr import is_ocr_available, get_ocr_info
print('OCR Available:', is_ocr_available())
print('OCR Info:', get_ocr_info())
"
```

### Test 2: Simple OCR Test

Create a test script `test_ocr.py`:

```python
from app.ocr import extract_text_from_image_file

# Replace with path to an image with text
result = extract_text_from_image_file('path/to/image.png')

print('Success:', result['success'])
print('Extracted Text:', result['text'])
if result.get('details'):
    print('Confidence Scores:', [d['confidence'] for d in result['details']])
```

Run it:
```powershell
python test_ocr.py
```

## 📚 Next Steps

After successful installation:

1. ✅ Read the **OCR_GUIDE.md** for detailed usage instructions
2. ✅ Test with sample images (screenshots, photos, scans)
3. ✅ Review extracted text accuracy
4. ✅ Adjust preprocessing settings if needed
5. ✅ Start using OCR with your documents!

## 🆘 Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review error messages carefully
3. Verify virtual environment is activated
4. Check Python version (must be 3.10+)
5. Ensure sufficient RAM (4GB minimum)
6. Check disk space (500MB free)
7. Try installing packages individually
8. Search for similar issues online

## 📖 Additional Resources

- **EasyOCR GitHub**: https://github.com/JaidedAI/EasyOCR
- **OpenCV Documentation**: https://docs.opencv.org/
- **NotesGPT README**: See README.md
- **OCR Usage Guide**: See OCR_GUIDE.md

---

**Installation Complete! Happy OCR-ing! 🎉**
