# OCR Text Recognition Application

This application uses EasyOCR to extract both computer-generated and handwritten text from images. It provides a user-friendly web interface to upload images or browse existing images in the Archive folder.

## Features

- Extract text from both computer-generated and handwritten documents
- Upload new images or process existing images from the Archive folder
- View detected text with confidence scores
- Visualize text locations with bounding boxes
- Copy extracted text to clipboard

## Requirements

- Python 3.7+
- EasyOCR
- Flask
- PyTorch
- Other dependencies (see requirements.txt)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- EasyOCR: For optical character recognition
- Flask: For the web interface
- Pillow: For image processing
- PyTorch: Required by EasyOCR
- Other dependencies

## Usage

1. Start the application:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://localhost:5000
```

3. Use the interface to:
   - Upload new images
   - Browse images in the Archive folder
   - Process images and view extracted text

## Notes

- The first time you run OCR on an image, EasyOCR will download the necessary language models (if not already downloaded)
- Processing time depends on image size and complexity
- The application currently supports English text by default

## Project Structure

- `app.py` - Main Flask application
- `ocr_processor.py` - OCR processing logic using EasyOCR
- `templates/` - HTML templates for the web interface
- `uploads/` - Directory for uploaded images
- `Archive/` - Directory for existing images


<!-- username:postgres
port: 5432
db: postgres
password:admin -->