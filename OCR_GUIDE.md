# 🔍 NotesGPT OCR Feature Guide

## Overview

The OCR (Optical Character Recognition) feature in NotesGPT allows you to extract text from images, scanned documents, and photos. This powerful feature uses EasyOCR, a state-of-the-art deep learning engine that supports multiple languages and provides high accuracy.

## 🌟 Key Features

### Multi-Language Support
- **English** (en)
- **French** (fr) 
- **Arabic** (ar)
- **Spanish** (es)
- **German** (de)
- **Italian** (it)
- **Portuguese** (pt)
- **Russian** (ru)
- **Chinese** (zh)
- **Japanese** (ja)
- **Korean** (ko)
- And many more...

### Advanced Image Preprocessing
- **Noise Reduction**: Bilateral filtering to remove noise while preserving edges
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Adaptive Thresholding**: Handles varying lighting conditions
- **Automatic Orientation**: Works with different image orientations

### User-Friendly Workflow
1. Upload image
2. AI extracts text automatically
3. Review and edit extracted text
4. Add to your library with one click

## 📋 Supported File Formats

- **PNG** (.png) - Recommended
- **JPEG** (.jpg, .jpeg)
- **TIFF** (.tiff, .tif)
- **BMP** (.bmp)
- **GIF** (.gif)
- **WebP** (.webp)

## 🚀 How to Use

### Step 1: Prepare Your Image

For best results:
- ✅ Use high-resolution images (300 DPI or higher)
- ✅ Ensure good lighting and contrast
- ✅ Keep text horizontal and properly aligned
- ✅ Avoid shadows, glare, or reflections
- ✅ Use clear, legible fonts
- ✅ Crop to focus on text area if possible

### Step 2: Upload Image

1. Navigate to the **"🔍 OCR - Extract Text from Images"** section
2. Click **"📷 Choose image..."**
3. Select your image file
4. Click **"🔍 Extract Text (OCR)"**

### Step 3: Processing

- First-time use: EasyOCR will download language models (50-100 MB per language)
- This only happens once and takes 1-3 minutes
- Subsequent OCR operations are much faster (10-30 seconds)

### Step 4: Review Results

After processing, you'll see:
- **Image Preview**: Visual confirmation of uploaded image
- **Extracted Text**: Text extracted from the image
- **Metadata**:
  - Filename
  - Word count
  - Confidence score (accuracy percentage)
  - Detected languages

### Step 5: Edit Text

- The extracted text appears in an editable text box
- Make any corrections needed
- Fix OCR errors or formatting
- Add missing punctuation or line breaks

### Step 6: Add to Library

- Click **"✅ Add to Library"**
- The text is indexed into your knowledge base
- You can now ask questions about the content
- The document appears in your library as "[filename] (OCR)"

## 💡 Use Cases

### 1. Scanned Documents
Convert scanned PDFs or photos of documents into searchable text:
- Research papers
- Textbooks
- Legal documents
- Medical records
- Receipts and invoices

### 2. Handwritten Notes
Extract text from handwritten notes (works best with clear handwriting):
- Class notes
- Meeting notes
- Journal entries
- To-do lists

### 3. Screenshots
Extract text from screenshots:
- Social media posts
- Website content
- Error messages
- Code snippets

### 4. Photos of Text
Process photos taken with your phone:
- Street signs
- Menu items
- Product labels
- Book pages
- Whiteboard content

### 5. Multi-Language Documents
Process documents in different languages:
- Foreign language textbooks
- International contracts
- Multilingual signs
- Translation sources

## 🎯 Best Practices

### Image Quality
- **Resolution**: Minimum 150 DPI, recommended 300+ DPI
- **Format**: PNG or TIFF for best quality (lossless)
- **Size**: 1-10 MB optimal, avoid very large files

### Lighting & Contrast
- Ensure even lighting across the document
- Avoid shadows from your phone or hand
- Use natural light or bright indoor lighting
- Increase contrast if text is faint

### Text Orientation
- Keep text horizontal
- Avoid skewed or rotated images
- Use image editing software to rotate if needed
- Crop unnecessary borders

### Document Preparation
- Flatten pages (no creases or folds)
- Place document on flat surface
- Remove glare from glossy paper
- Clean dirty or stained pages if possible

### Batch Processing
- Process one image at a time for accuracy
- Group similar documents together
- Name files descriptively
- Review each extraction before indexing

## ⚙️ Technical Details

### OCR Engine
- **Engine**: EasyOCR (v1.7.1)
- **Backend**: PyTorch deep learning models
- **Languages**: Configurable, default: en, fr, ar
- **GPU Support**: Optional (edit `app/ocr.py`)

### Image Processing Pipeline
1. **Load Image**: PIL/OpenCV
2. **Convert**: RGB to BGR if needed
3. **Preprocess**: 
   - Grayscale conversion
   - Bilateral filter (noise reduction)
   - CLAHE (contrast enhancement)
   - Adaptive thresholding
4. **OCR**: EasyOCR text detection and recognition
5. **Post-process**: Format and combine text segments

### Confidence Scores
- Each detected text segment has a confidence score (0-1)
- Average confidence is displayed to user
- Higher confidence = better accuracy
- Typical good confidence: 0.7-0.95 (70-95%)

### Performance
- **First Run**: 1-3 minutes (model download)
- **Subsequent Runs**: 10-30 seconds (CPU)
- **With GPU**: 2-5 seconds
- **Memory Usage**: 500MB - 2GB RAM

## 🔧 Advanced Configuration

### Changing Languages

Edit `app/ocr.py` to change default languages:

```python
# Default languages for OCR
DEFAULT_LANGUAGES = ['en', 'fr', 'ar']  # Modify this list
```

Available language codes:
- `en` - English
- `fr` - French
- `ar` - Arabic
- `es` - Spanish
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean

### Enabling GPU Acceleration

If you have an NVIDIA GPU with CUDA:

1. Install CUDA toolkit from NVIDIA website
2. Install PyTorch with CUDA support
3. Edit `app/ocr.py`:

```python
def get_ocr_reader(languages: List[str] = None) -> Optional[Any]:
    # Change gpu=False to gpu=True
    _ocr_reader = easyocr.Reader(languages, gpu=True)  # Enable GPU
```

### Preprocessing Options

Edit `preprocess_image()` function in `app/ocr.py` to customize:

```python
# Adjust bilateral filter parameters
denoised = cv2.bilateralFilter(gray, 9, 75, 75)  # Modify these values

# Adjust CLAHE parameters
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Modify these

# Adjust adaptive threshold
threshold = cv2.adaptiveThreshold(
    enhanced,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,  # Block size
    2    # Constant
)
```

## 🐛 Troubleshooting

### OCR Not Working

**Problem**: "OCR functionality not available" error

**Solution**:
```bash
pip install easyocr opencv-python
```

### Slow Performance

**Problem**: OCR is very slow

**Solutions**:
- First run downloads models (one-time delay)
- Use smaller images (resize to 1500px max width)
- Enable GPU acceleration if available
- Close other memory-intensive applications
- Process one image at a time

### Poor Accuracy

**Problem**: Extracted text has many errors

**Solutions**:
- Improve image quality (higher resolution)
- Increase contrast before uploading
- Ensure text is horizontal and in focus
- Remove noise and artifacts
- Try different image formats
- Use proper lighting when photographing

### Low Confidence Scores

**Problem**: Confidence scores below 50%

**Solutions**:
- Check image quality
- Verify text is clearly visible
- Ensure language is correctly set
- Try preprocessing image externally
- Use higher resolution source

### Out of Memory

**Problem**: System runs out of RAM

**Solutions**:
- Reduce image size (max 2000x2000 pixels)
- Close other applications
- Process images one at a time
- Restart the application
- Increase system RAM if possible

### Text Not Detected

**Problem**: OCR returns "No text detected"

**Solutions**:
- Verify image contains visible text
- Check text is not too small (min 12pt)
- Ensure adequate contrast
- Try inverting colors (white text on black)
- Check text language is supported

## 📊 Performance Tips

### Optimize Image Size
```python
# Before uploading, resize large images:
from PIL import Image

img = Image.open('large_image.jpg')
img.thumbnail((2000, 2000))  # Max 2000px on longest side
img.save('optimized_image.jpg', quality=90)
```

### Batch Processing Workflow
1. Prepare all images in advance
2. Optimize and crop each image
3. Process during off-peak hours
4. Review and edit in batches
5. Index after all corrections

### Memory Management
- Process images one at a time
- Clear browser cache regularly
- Restart application after 20-30 OCR operations
- Monitor system memory usage

## 🎓 Examples

### Example 1: Scanned Textbook Page

**Input**: Photo of textbook page
**Steps**:
1. Take photo with good lighting
2. Crop to text area
3. Upload to OCR
4. Review extracted text
5. Fix any formatting issues
6. Add to library

**Result**: Searchable textbook content

### Example 2: Handwritten Notes

**Input**: Photo of handwritten notes
**Steps**:
1. Ensure neat handwriting
2. Use high-contrast paper
3. Photograph directly from above
4. Upload to OCR
5. Carefully review and edit
6. Add to library

**Result**: Searchable note content

### Example 3: Multi-Language Document

**Input**: Document with English and French text
**Steps**:
1. Ensure languages are in DEFAULT_LANGUAGES
2. Upload document
3. OCR detects both languages
4. Review mixed-language text
5. Add to library

**Result**: Searchable multilingual content

## 🔐 Privacy & Security

- All OCR processing happens locally (no external API calls)
- Images are temporarily stored in `uploads/` directory
- Extracted text is stored in your local vector database
- No data is sent to external servers (except Groq for chat)
- Delete sensitive documents after indexing if needed

## 📚 Additional Resources

- **EasyOCR Documentation**: https://github.com/JaidedAI/EasyOCR
- **OpenCV Documentation**: https://opencv.org/
- **NotesGPT Issues**: Report bugs on GitHub

## 🆘 Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review error messages carefully
3. Verify all dependencies are installed
4. Check system requirements are met
5. Try with a different image
6. Report bugs with example images and error logs

---

**Happy OCR-ing! 📸✨**
