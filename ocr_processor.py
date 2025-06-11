import easyocr
import numpy as np
import os
import time
import cv2
import tempfile
import re
from PIL import Image, ImageOps
from PyPDF2 import PdfReader
from ai_analyzer import AIAnalyzer

class OCRProcessor:
    def __init__(self):
        # Initialize the EasyOCR reader with languages
        # English is default, can be expanded to other languages as needed
        self._reader = None
        self.languages = ['en']
        
        # Initialize the AI analyzer
        self._ai_analyzer = AIAnalyzer()
        
    def _load_reader(self):
        """Lazy loading of the EasyOCR reader to save resources"""
        if self._reader is None:
            print("Loading EasyOCR model... This may take a moment.")
            self._reader = easyocr.Reader(self.languages)
        return self._reader
            
    def process_image(self, image_path, reading_direction='default'):
        """
        Process an image or PDF file and extract text using EasyOCR
        
        Args:
            image_path (str): Path to the image or PDF file
            reading_direction: 'default', 'horizontal', or 'vertical' to control text ordering
            
        Returns:
            dict: Dictionary containing results and processing information
        """
        try:
            start_time = time.time()
            file_extension = os.path.splitext(image_path)[1].lower()
            
            # Check if the file is a PDF
            if file_extension == '.pdf':
                return self._process_pdf(image_path, reading_direction)
                
            # Load and validate the image directly with OpenCV to avoid resize errors
            try:
                # Load image directly with OpenCV instead of letting EasyOCR handle it
                cv_img = cv2.imread(image_path)
                
                # Debug information
                print(f"Image loaded from path: {image_path}")
                if cv_img is None:
                    print("OpenCV image loading failed: cv_img is None")
                    return {
                        'success': False,
                        'error': 'Failed to load image with OpenCV - image is None',
                        'processing_time': time.time() - start_time
                    }
                elif cv_img.size == 0:
                    print("OpenCV image loading failed: cv_img size is 0")
                    return {
                        'success': False,
                        'error': 'Failed to load image with OpenCV - image size is 0',
                        'processing_time': time.time() - start_time
                    }
                
                # Get image dimensions and print debug info
                img_height, img_width = cv_img.shape[:2]
                print(f"Image dimensions: {img_width}x{img_height}")
                
                # If image is too small or has invalid dimensions, resize it to ensure minimum dimensions
                min_dimension = 32  # Most CV operations need at least some minimal size
                if img_width < min_dimension or img_height < min_dimension:
                    print(f"Image too small, resizing to minimum dimensions")
                    scale_factor = max(min_dimension / img_width, min_dimension / img_height)
                    new_width = max(min_dimension, int(img_width * scale_factor))
                    new_height = max(min_dimension, int(img_height * scale_factor))
                    cv_img = cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    img_height, img_width = cv_img.shape[:2]
                    print(f"Resized image dimensions: {img_width}x{img_height}")
                
                # Also open with PIL for compatibility with rest of the code
                try:
                    img = Image.open(image_path)
                    print(f"PIL Image dimensions: {img.width}x{img.height}")
                except Exception as pil_error:
                    print(f"PIL image loading failed but OpenCV succeeded: {str(pil_error)}")
                    # Convert OpenCV image to PIL format as fallback
                    img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                    print(f"Created PIL image from OpenCV with dimensions: {img.width}x{img.height}")
            except Exception as img_error:
                print(f'Error loading image: {str(img_error)}')
                return {
                    'success': False,
                    'error': f'Image loading error: {str(img_error)}',
                    'processing_time': time.time() - start_time
                }
            
            # Get the reader
            reader = self._load_reader()
            
            # Extract text from the image using our pre-loaded OpenCV image
            # instead of letting EasyOCR load it internally
            try:
                # Pass the OpenCV image directly to avoid internal resize issues
                results = reader.readtext(cv_img, detail=1)
            except Exception as ocr_error:
                print(f'Error during OCR text reading: {str(ocr_error)}')
                return {
                    'success': False,
                    'error': f'OCR processing error: {str(ocr_error)}',
                    'processing_time': time.time() - start_time
                }
            
            # Process results
            extracted_text = []
            for detection in results:
                bbox, text, confidence = detection
                # Convert NumPy values to native Python types
                converted_bbox = []
                for point in bbox:
                    converted_bbox.append([int(point[0]), int(point[1])])
                
                extracted_text.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': converted_bbox
                })
            
            # Sort by confidence score (descending)
            extracted_text.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Sort text based on reading direction if specified
            if reading_direction == 'horizontal':
                # Sort by y-coordinate (top to bottom) and then by x-coordinate (left to right) within same line
                extracted_text.sort(key=lambda x: (x['bbox'][0][1] + x['bbox'][2][1]) / 2)  # Sort by average y-coordinate
                
                # Group text elements by lines (similar y-coordinate)
                line_groups = []
                current_group = [extracted_text[0]]
                y_threshold = img.height * 0.02  # 2% of image height as threshold for same line
                
                for i in range(1, len(extracted_text)):
                    current_y = (current_group[0]['bbox'][0][1] + current_group[0]['bbox'][2][1]) / 2
                    next_y = (extracted_text[i]['bbox'][0][1] + extracted_text[i]['bbox'][2][1]) / 2
                    
                    if abs(next_y - current_y) < y_threshold:  # Same line
                        current_group.append(extracted_text[i])
                    else:  # New line
                        # Sort current group by x-coordinate (left to right)
                        current_group.sort(key=lambda x: x['bbox'][0][0])
                        line_groups.append(current_group)
                        current_group = [extracted_text[i]]
                
                # Add the last group
                current_group.sort(key=lambda x: x['bbox'][0][0])
                line_groups.append(current_group)
                
                # Flatten the groups back
                extracted_text = [item for group in line_groups for item in group]
                
                print("Applied horizontal reading direction sorting")
            
            # Get the full text as a string
            full_text = self.get_full_text({'success': True, 'results': extracted_text})
            
            # Analyze the text with AI
            ai_analysis = self._analyze_text(full_text)
            print(f"AI Analysis Result: {ai_analysis}") # Add debug logging
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'results': extracted_text,
                'processing_time': processing_time,
                'image_size': (img.width, img.height),
                'filename': os.path.basename(image_path),
                'full_text': full_text,
                'ai_analysis': ai_analysis
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            
    def _process_pdf(self, pdf_path, reading_direction='default'):
        """
        Process a PDF file by extracting text directly using PyPDF2
        
        Args:
            pdf_path (str): Path to the PDF file
            reading_direction: Not used for direct PDF text extraction
            
        Returns:
            dict: Dictionary containing results and processing information
        """
        start_time = time.time()
        
        try:
            print(f"Extracting text from PDF {pdf_path}...")
            
            # Attempt to open the PDF file with more robust error handling
            try:
                # First try standard PDF reading
                pdf = PdfReader(pdf_path)
                num_pages = len(pdf.pages)
            except Exception as pdf_error:
                print(f"Standard PDF reading failed: {str(pdf_error)}")
                print("Attempting fallback to image-based OCR processing...")
                
                # Fallback 1: Try PyMuPDF (fitz) for PDF reading
                try:
                    import fitz  # PyMuPDF as a first fallback alternative
                    print("Using PyMuPDF (fitz) as fallback for PDF reading")
                    
                    try:
                        # Open the PDF file with PyMuPDF
                        doc = fitz.open(pdf_path)
                        num_pages = doc.page_count
                        
                        if num_pages > 0:
                            # Process first page as image to demonstrate it works
                            page = doc[0]
                            pix = page.get_pixmap()
                            img_path = f"{pdf_path}_page0.png"
                            pix.save(img_path)
                            
                            # Process this image with OCR
                            ocr_result = self.process_image(img_path, reading_direction)
                            ocr_result['is_pdf'] = True
                            ocr_result['num_pages'] = num_pages
                            ocr_result['was_processed_as_image'] = True
                            
                            # Clean up the temporary image
                            try:
                                os.remove(img_path)
                            except:
                                pass
                                
                            return ocr_result
                    except Exception as pymupdf_err:
                        print(f"PyMuPDF fallback failed: {str(pymupdf_err)}")
                        raise  # Let it go to the next fallback
                except Exception:
                    # Fallback 2: Try direct image processing (treat PDF as an image)
                    print("Attempting extreme fallback - treating PDF directly as an image file")
                    try:
                        from PIL import Image
                        import shutil
                        
                        # Create a temporary copy with .jpg extension
                        temp_img_path = pdf_path + ".jpg"
                        shutil.copy(pdf_path, temp_img_path)
                        
                        try:
                            # Try to process it as a regular image
                            result = self.process_image(temp_img_path, reading_direction)
                            # Only mark as successful if the result actually succeeded
                            if result.get('success', False):
                                print("Successfully processed corrupted PDF as image")
                                result['is_pdf'] = True  # Still indicate it was a PDF originally
                                result['was_processed_as_image'] = True
                                result['pdf_was_corrupted'] = True
                            else:
                                # If process_image failed, propagate the error message
                                print(f"Image-based processing failed: {result.get('error', 'Unknown error')}")
                                raise Exception(result.get('error', 'Failed to process as image'))
                            
                            # Remove temporary file
                            try:
                                os.remove(temp_img_path)
                            except:
                                pass
                                
                            return result
                        except Exception as img_err:
                            print(f"Direct image treatment failed: {str(img_err)}")
                            os.remove(temp_img_path)
                            
                            # One more attempt - extract embedded images if any
                            try:
                                with open(pdf_path, 'rb') as f:
                                    content = f.read()
                                    
                                # Look for JPEG markers in the binary content
                                jpg_markers = [b'\xff\xd8\xff', b'JFIF', b'Exif']
                                png_markers = [b'\x89PNG']
                                
                                has_images = any(marker in content for marker in jpg_markers + png_markers)
                                
                                if has_images:
                                    img_output = f"{pdf_path}_extracted.jpg"
                                    with open(img_output, 'wb') as out_f:
                                        # Write the part of the file that might contain image data
                                        out_f.write(content)
                                        
                                    try:
                                        result = self.process_image(img_output, reading_direction)
                                        print("Processed extracted image content from corrupted PDF")
                                        return result
                                    except:
                                        pass
                                    finally:
                                        try:
                                            os.remove(img_output)
                                        except:
                                            pass
                            except:
                                pass                            
                    except Exception as final_err:
                        print(f"All PDF recovery attempts failed: {str(final_err)}")
                    
                    # If all fallbacks fail, return a friendly error message
                    return {
                        'success': False,
                        'error': 'PDF file appears to be severely corrupted or damaged. Please try uploading a different file or repairing the PDF first.',
                        'file_path': pdf_path,  # Include path for reference
                        'processing_time': time.time() - start_time
                    }
            
            # If we get here, we successfully opened the PDF with standard method
            if num_pages == 0:
                return {
                    'success': False,
                    'error': 'PDF file has no pages',
                    'processing_time': time.time() - start_time
                }
                
            # Extract text from each page
            all_results = []
            full_text = ""
            page_heights = []  # Store page heights for reference
            page_widths = []   # Store page widths for reference
            
            # Get metadata (size from first page - approximate)
            # PyPDF2 doesn't directly provide page size, but we'll estimate
            try:
                # Try to get page size if available in the PDF
                media_box = pdf.pages[0].mediabox
                width = float(media_box[2] - media_box[0])
                height = float(media_box[3] - media_box[1])
            except:
                # Use default values if not available
                width = 612.0  # Standard letter width in points
                height = 792.0 # Standard letter height in points
                
            for i in range(num_pages):
                page = pdf.pages[i]
                page_text = page.extract_text() or ""
                
                if page_text.strip():
                    # Clean up the text (remove extra whitespace)
                    page_text = re.sub(r'\s+', ' ', page_text).strip()
                    
                    # Add page text to full text
                    full_text += page_text + "\n\n"
                    
                    # Split the page text into lines
                    lines = page_text.split('\n')
                    
                    # Process each line as a separate text element
                    # Since we don't have bbox info, we'll create approximate ones
                    line_height = height / max(len(lines), 1)
                    
                    for j, line in enumerate(lines):
                        if line.strip():
                            # Create synthetic bbox (we don't have real coordinates)
                            # Just for compatibility with the rest of the app
                            y_pos = j * line_height
                            box_height = line_height * 0.8
                            
                            synthetic_bbox = [
                                [10, y_pos], 
                                [width-10, y_pos],
                                [width-10, y_pos + box_height],
                                [10, y_pos + box_height]
                            ]
                            
                            # Add to results
                            all_results.append({
                                'text': line,
                                'confidence': 1.0,  # We assume high confidence for direct extraction
                                'bbox': synthetic_bbox,
                                'page': i+1
                            })
            
            # Run AI analysis on the extracted text
            ai_analysis = self._analyze_text(full_text)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'results': all_results,
                'processing_time': processing_time,
                'image_size': (int(width), int(height)),
                'filename': os.path.basename(pdf_path),
                'full_text': full_text,
                'ai_analysis': ai_analysis,
                'is_pdf': True,
                'num_pages': num_pages,
                'is_searchable_pdf': True  # Flag to indicate text was directly extracted
            }
        
        except Exception as e:
            import traceback
            print(f"Error processing PDF: {str(e)}")
            print(traceback.format_exc())
            return {
                'success': False,
                'error': f"PDF processing error: {str(e)}",
                'processing_time': time.time() - start_time
            }
    
    def get_full_text(self, results):
        """
        Extract just the text from OCR results
        
        Args:
            results (dict): OCR processing results
            
        Returns:
            str: All extracted text concatenated
        """
        if not results.get('success', False):
            return "Error: " + results.get('error', 'Unknown error')
        
        # Extract and join all text items
        text_items = [item['text'] for item in results.get('results', [])]
        return '\n'.join(text_items)
        
    def _analyze_text(self, text):
        """
        Analyze the extracted text with AI to extract structured information
        
        Args:
            text (str): The full extracted text from OCR
            
        Returns:
            dict: Structured information extracted by AI
        """
        try:
            # Use the AI analyzer to extract structured information
            analysis_result = self._ai_analyzer.analyze_text(text)
            return analysis_result
        except Exception as e:
            print(f"AI Analysis error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
