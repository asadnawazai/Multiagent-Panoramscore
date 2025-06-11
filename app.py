import eventlet
eventlet.monkey_patch()

import sys
import os
import re
import uuid
import json
import yaml
import glob
import time
import base64
import hashlib
import logging
import datetime
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ocr_processor import OCRProcessor
from db_service import DatabaseService
from flask_socketio import SocketIO, emit, join_room, leave_room
from socket_events import (
    send_upload_progress, send_processing_status, send_missing_fields_update,
    send_field_updated, send_process_complete, send_error
)
import numpy as np

app = Flask(__name__)

# Configure CORS - using array format for origins as developer requested
CORS(app, resources={r"/*": {"origins": ["*"]}})


# Configure SocketIO with CORS support - using array format for origins
socketio = SocketIO(
    app,
    cors_allowed_origins=["*"],      # Allow all origins in array format
    async_mode='eventlet',           # Switch to eventlet for better proxy support
    path='/socket.io',               # Explicitly set path
    logger=True,
    engineio_logger=True,
    ping_timeout=60000,              # Increased timeout for stability
    ping_interval=25000,             # Increased interval
    manage_session=False,            # Let Flask handle sessions
    transports=['polling', 'websocket'],  # Start with polling, upgrade to websocket
    allow_upgrades=True,            # Enable transport upgrades
    always_connect=True             # Always allow connections
)

# Dictionary to track active session rooms
active_sessions = {}

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    
@socketio.on('join')
def on_join(data):
    """Client joins a room based on session ID"""
    if 'session_id' in data:
        session_id = data['session_id']
        join_room(session_id)
        active_sessions[session_id] = True
        print(f'Client joined room: {session_id}')
        emit('joined', {'session_id': session_id, 'status': 'connected'})
    else:
        emit('error', {'message': 'No session_id provided'})

@socketio.on('leave')
def on_leave(data):
    """Client leaves a room"""
    if 'session_id' in data:
        session_id = data['session_id']
        leave_room(session_id)
        if session_id in active_sessions:
            del active_sessions[session_id]
        print(f'Client left room: {session_id}')
        emit('left', {'session_id': session_id, 'status': 'disconnected'})
    else:
        emit('error', {'message': 'No session_id provided'})

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ARCHIVE_FOLDER'] = 'Archive'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff', 'xlsx', 'csv'}

# Create required folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ARCHIVE_FOLDER'], exist_ok=True)

# Initialize OCR processor
ocr_processor = OCRProcessor()

# Initialize database service
db_service = DatabaseService()

# Ensure database table exists
db_service.connect()
db_service.create_table_if_not_exists()

@app.teardown_appcontext
def close_db_connection(exception=None):
    """Close the database connection at the end of each request"""
    db_service.close()

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_excel_csv_file(filepath):
    """Process Excel (.xlsx) or CSV file to extract data"""
    file_ext = os.path.splitext(filepath)[1].lower()
    data = {}
    
    try:
        if file_ext == '.xlsx':
            # Read Excel file
            df = pd.read_excel(filepath)
        elif file_ext == '.csv':
            # Read CSV file
            df = pd.read_csv(filepath)
        else:
            return {"success": False, "error": "Unsupported file format"}
        
        # Convert DataFrame to dictionary
        if not df.empty:
            # Create a text representation of the data for embedding
            text_content = []
            
            # Extract field values if columns match expected fields
            extracted_data = {}
            for col in df.columns:
                # Clean up column names (strip whitespace, case-insensitive comparison)
                clean_col = col.strip()
                
                # Check for each possible field
                for field in ["Build Year", "Climate Score", "Flood Risk Score", 
                             "Infrastructure Opacity", "Land Use Code", "Mls Listing",
                             "Outdated Tax Delta", "Regional Data Variation", "Zoning Record",
                             "Property Address", "Owner Name", "Parcel Number",
                             "Property Type & Use", "Legal Description"]:
                    if clean_col.lower() == field.lower() or clean_col == field:
                        # Get first non-empty value from the column
                        values = df[col].dropna()
                        if not values.empty:
                            value = str(values.iloc[0])
                            extracted_data[field] = value
                            text_content.append(f"{field}: {value}")
            
            # Create full text content for embedding
            full_text = "\n".join(text_content) if text_content else ""
            
            # Format response like OCR processor output
            return {
                "success": True,
                "filename": os.path.basename(filepath),
                "ai_analysis": {
                    "analyzed_data": extracted_data
                },
                "full_text": full_text
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    return {"success": False, "error": "No data found"}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process', methods=['POST'])
def process_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Create a unique filename to avoid collisions
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        new_filename = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        
        # Ensure upload folder exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Process the file based on type
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Determine if the file is a PDF - define this before using it later
        is_pdf = file_ext == '.pdf'
        
        # Initialize OCR processor if needed
        ocr_processor = OCRProcessor()
        
        try:
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                results = ocr_processor.process_image(filepath)
            elif is_pdf:
                try:
                    # First attempt using the default OCR processor's PDF handling
                    results = ocr_processor.process_image(filepath)
                    if not results.get('success', False):
                        # If that fails, try direct PDF text extraction with PyPDF2
                        try:
                            from PyPDF2 import PdfReader
                            import fitz  # PyMuPDF
                            
                            # Try to extract text with PyPDF2
                            pdf = PdfReader(filepath)
                            full_text = ''
                            for page in pdf.pages:
                                full_text += page.extract_text() + '\n'
                                
                            if full_text.strip():
                                # If we got text, create a successful result
                                print('Successfully extracted text directly from PDF using PyPDF2')
                                results = {
                                    'success': True,
                                    'full_text': full_text,
                                    'is_pdf': True,
                                    'filename': os.path.basename(filepath),
                                    'processing_time': 0
                                }
                            else:
                                # Try with PyMuPDF as a secondary fallback
                                try:
                                    doc = fitz.open(filepath)
                                    full_text = ''
                                    for page in doc:
                                        full_text += page.get_text() + '\n'
                                        
                                    if full_text.strip():
                                        print('Successfully extracted text from PDF using PyMuPDF')
                                        results = {
                                            'success': True,
                                            'full_text': full_text,
                                            'is_pdf': True,
                                            'filename': os.path.basename(filepath),
                                            'processing_time': 0
                                        }
                                    else:
                                        raise Exception('No text extracted from PDF')
                                except Exception as mupdf_err:
                                    print(f'PyMuPDF extraction failed: {str(mupdf_err)}')
                                    raise Exception('PDF text extraction failed with all methods')
                        except Exception as direct_err:
                            print(f'Direct PDF extraction failed: {str(direct_err)}')
                            raise Exception(results.get('error', 'PDF processing failed with all available methods'))
                except Exception as pdf_err:
                    print(f'File processing failed with error: {str(pdf_err)}')
                    return jsonify({
                        'success': False,
                        'error': f'Could not process PDF file: {str(pdf_err)}. Please try a different file format or repair the PDF.'
                    })
            else:
                return jsonify({
                    'success': False, 
                    'error': f'Unsupported file type: {file_ext}'
                })
            
            # Check if we got a successful result or an error
            if not results.get('success', True):
                # Return the error to the frontend with a user-friendly message
                error_msg = results.get('error', 'Unknown error processing file')
                print(f"File processing failed with error: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'description': 'The uploaded file appears to be corrupted or invalid. Please try a different file.'
                })
        except Exception as e:
            print(f"Exception during file processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Error processing file: {str(e)}',
                'description': 'An unexpected error occurred during processing. Please try again or use a different file.'
            })

        # If OCR was successful, save file to archive
        if results['success']:
            # Copy file to archive
            archive_path = os.path.join(app.config['ARCHIVE_FOLDER'], new_filename)
            shutil.copy2(filepath, archive_path)
            
            # Save results to JSON for later retrieval
            json_path = os.path.join(app.config['ARCHIVE_FOLDER'], f"{os.path.splitext(new_filename)[0]}.json")
            with open(json_path, 'w') as f:
                json.dump(results, f)

            # Add file URL for frontend display - use new_filename which contains the UUID
            results['file_url'] = f"/uploads/{new_filename}"
            
            # Add PDF flag if applicable
            if is_pdf:
                results['is_pdf'] = True
                
            # Store results in the database if AI analysis was successful
            similar_documents = []
            if results['success'] and 'ai_analysis' in results and results['ai_analysis']['success']:
                try:
                    print(f"Attempting to store document: {filepath}")
                    # Ensure full_text is properly formatted before sending to the database
                    # We want to ensure the complete extracted text shown on frontend is stored in DB
                    full_text = results.get('full_text', '')
                    if full_text:
                        print(f"Preparing full_text for storage, length: {len(full_text)}")
                        # Convert to string explicitly in case it's not already a string
                        full_text = str(full_text)
                    else:
                        print("Warning: full_text is empty")
                        # If full_text is empty, try to get complete_text from analyzed_data
                        full_text = results['ai_analysis']['analyzed_data'].get('complete_text', '')
                        if full_text:
                            print(f"Using complete_text instead, length: {len(full_text)}")
                            full_text = str(full_text)
                    
                    # Make sure the text isn't truncated before storage
                    print(f"Final text for DB storage, first 100 chars: {full_text[:100]}")
                    print(f"Final text for DB storage, last 100 chars: {full_text[-100:] if len(full_text) > 100 else full_text}")
                    
                    stored = db_service.store_document(
                        file_path=filepath,
                        analyzed_data=results['ai_analysis']['analyzed_data'],
                        extracted_text=full_text,
                        force_store=True  # Force storage even if other fields are missing
                    )
                    print(f"Document stored successfully: {stored}")
                    
                    try:
                        # Get embeddings for the document
                        print(f"Generating embeddings for document")
                        embedding = db_service._get_embedding(results['full_text'])
                        
                        if embedding:
                            # Find similar documents using vector similarity search
                            print(f"Finding similar documents using vector similarity search")
                            similar_documents = db_service.find_similar_documents(embedding, limit=3)
                            print(f"Found {len(similar_documents)} similar documents")
                            
                            # Add similar documents to the result
                            results['similar_documents'] = similar_documents
                        else:
                            print("Warning: Empty embedding vector generated")
                            results['similar_documents'] = []
                    except Exception as e:
                        print(f"Error during embedding or similarity search: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        results['similar_documents'] = []
                except Exception as e:
                    print(f"Error storing document in database: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
        elif is_pdf and "poppler" in str(results.get('error', '')).lower():
            # Special handling for PDF poppler missing error
            results['is_pdf'] = True
            results['needs_poppler'] = True
        
        return jsonify(results)
    
    return jsonify({'success': False, 'error': 'File type not allowed'})

@app.route('/browse')
def browse_archive():
    """Browse images in the Archive folder"""
    archive_path = app.config['ARCHIVE_FOLDER']
    files = []
    
    # List files in the Archive directory
    for filename in os.listdir(archive_path):
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if ext in app.config['ALLOWED_EXTENSIONS']:
            files.append({
                'filename': filename,
                'path': os.path.join(archive_path, filename)
            })
    
    return render_template('browse.html', files=files)

@app.route('/process-archive', methods=['POST'])
def process_archive_image():
    """Process an image from the Archive folder"""
    data = request.json
    if not data or 'filepath' not in data:
        return jsonify({'success': False, 'error': 'No file path provided'})
    
    filepath = data['filepath']
    
    # Verify the file exists and is in the Archive folder
    if not os.path.exists(filepath) or not filepath.startswith('/Users/gsoft1234/Desktop/OCR/Archive'):
        return jsonify({'success': False, 'error': 'Invalid file path'})
    
    # Process the image with OCR using horizontal reading direction
    results = ocr_processor.process_image(filepath, reading_direction='horizontal')
    
    # Add the file info to the results
    if results['success']:
        results['filename'] = os.path.basename(filepath)
    
    return jsonify(results)

@app.route('/update-document', methods=['POST'])
def update_document():
    """Update document with user-filled values and store in database"""
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'})
    
    # Extract required fields
    file_path = data.get('file_path')
    analyzed_data = data.get('analyzed_data', {})
    extracted_text = data.get('extracted_text', '')
    
    if not file_path or not analyzed_data or not extracted_text:
        return jsonify({'success': False, 'error': 'Missing required fields'})
    
    try:
        # Clear any 'not found' values that might still be in the data
        for key, value in analyzed_data.items():
            if value == 'not found':
                analyzed_data[key] = ''
                
        # Force store in database even if some fields still have 'not found' values
        print(f"Force storing document: {file_path}")
        
        # Ensure extracted_text is properly formatted before sending to database
        if extracted_text:
            print(f"Update endpoint: Preparing extracted_text for storage, length: {len(extracted_text)}")
        else:
            print("Warning: extracted_text is empty in update endpoint")
            # If we have complete_text in analyzed_data, use that instead
            extracted_text = analyzed_data.get('complete_text', '')
            if extracted_text:
                print(f"Update endpoint: Using complete_text instead, length: {len(extracted_text)}")
                
        stored = db_service.store_document(file_path, analyzed_data, extracted_text, force_store=True)
        print(f"Document stored successfully: {stored}")
        
        similar_documents = []
        try:
            # Generate embeddings and find similar documents
            print(f"Generating embeddings for updated document")
            embedding = db_service._get_embedding(extracted_text)
            
            if embedding:
                print(f"Finding similar documents for updated document")
                similar_documents = db_service.find_similar_documents(embedding, limit=3)
                print(f"Found {len(similar_documents)} similar documents for updated document: {similar_documents}")
            else:
                print("Warning: Empty embedding vector generated for updated document")
        except Exception as e:
            print(f"Error during embedding or similarity search for updated document: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return jsonify({
            'success': True, 
            'message': 'Document updated and stored in database successfully',
            'similar_documents': similar_documents
        })
        
    except Exception as e:
        print(f"Error updating document in database: {str(e)}")
        return jsonify({'success': False, 'error': f'Database error: {str(e)}'})


@app.route('/api/process', methods=['POST'])
def api_process_file():
    """API endpoint for file processing that can be tested with Postman"""
    # Check if file was uploaded correctly
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Create a unique filename to avoid collisions
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        new_filename = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        
        # Ensure upload folder exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Process the file based on type
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Determine if the file is a PDF
        is_pdf = file_ext == '.pdf'
        
        # Initialize OCR processor
        ocr_processor = OCRProcessor()
        
        try:
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                results = ocr_processor.process_image(filepath)
            elif is_pdf:
                try:
                    # First attempt using the default OCR processor's PDF handling
                    results = ocr_processor.process_image(filepath)
                    if not results.get('success', False):
                        # If that fails, try direct PDF text extraction with PyPDF2
                        try:
                            from PyPDF2 import PdfReader
                            import fitz  # PyMuPDF
                            
                            # Try to extract text with PyPDF2
                            pdf = PdfReader(filepath)
                            full_text = ''
                            for page in pdf.pages:
                                full_text += page.extract_text() + '\n'
                                
                            if full_text.strip():
                                # If we got text, create a successful result
                                print('Successfully extracted text directly from PDF using PyPDF2')
                                results = {
                                    'success': True,
                                    'full_text': full_text,
                                    'is_pdf': True,
                                    'filename': os.path.basename(filepath),
                                    'processing_time': 0
                                }
                            else:
                                # Try with PyMuPDF as a secondary fallback
                                try:
                                    doc = fitz.open(filepath)
                                    full_text = ''
                                    for page in doc:
                                        full_text += page.get_text() + '\n'
                                        
                                    if full_text.strip():
                                        print('Successfully extracted text from PDF using PyMuPDF')
                                        results = {
                                            'success': True,
                                            'full_text': full_text,
                                            'is_pdf': True,
                                            'filename': os.path.basename(filepath),
                                            'processing_time': 0
                                        }
                                    else:
                                        raise Exception('No text extracted from PDF')
                                except Exception as mupdf_err:
                                    print(f'PyMuPDF extraction failed: {str(mupdf_err)}')
                                    raise Exception('PDF text extraction failed with all methods')
                        except Exception as direct_err:
                            print(f'Direct PDF extraction failed: {str(direct_err)}')
                            raise Exception(results.get('error', 'PDF processing failed with all available methods'))
                except Exception as pdf_err:
                    print(f'File processing failed with error: {str(pdf_err)}')
                    return jsonify({
                        'success': False,
                        'error': f'Could not process PDF file: {str(pdf_err)}. Please try a different file format or repair the PDF.'
                    })
            else:
                return jsonify({
                    'success': False, 
                    'error': f'Unsupported file type: {file_ext}'
                })
            
            # Check if we got a successful result or an error
            if not results.get('success', True):
                # Return the error to the frontend with a user-friendly message
                error_msg = results.get('error', 'Unknown error processing file')
                print(f"File processing failed with error: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'description': 'The uploaded file appears to be corrupted or invalid. Please try a different file.'
                })
        except Exception as e:
            print(f"Exception during file processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Error processing file: {str(e)}',
                'description': 'An unexpected error occurred during processing. Please try again or use a different file.'
            })

        # If OCR was successful, save file to archive
        if results['success']:
            # Copy file to archive
            archive_path = os.path.join(app.config['ARCHIVE_FOLDER'], new_filename)
            shutil.copy2(filepath, archive_path)
            
            # Save results to JSON for later retrieval
            json_path = os.path.join(app.config['ARCHIVE_FOLDER'], f"{os.path.splitext(new_filename)[0]}.json")
            with open(json_path, 'w') as f:
                json.dump(results, f)

            # Add file URL for frontend display - use new_filename which contains the UUID
            results['file_url'] = f"/uploads/{new_filename}"
            
            # Add PDF flag if applicable
            if is_pdf:
                results['is_pdf'] = True
                
            # Store results in the database if AI analysis was successful
            similar_documents = []
            if results['success'] and 'ai_analysis' in results and results['ai_analysis']['success']:
                try:
                    print(f"Attempting to store document: {filepath}")
                    # Ensure full_text is properly formatted before sending to the database
                    # We want to ensure the complete extracted text shown on frontend is stored in DB
                    full_text = results.get('full_text', '')
                    if full_text:
                        print(f"Preparing full_text for storage, length: {len(full_text)}")
                        # Convert to string explicitly in case it's not already a string
                        full_text = str(full_text)
                    else:
                        print("Warning: full_text is empty")
                        # If full_text is empty, try to get complete_text from analyzed_data
                        full_text = results['ai_analysis']['analyzed_data'].get('complete_text', '')
                        if full_text:
                            print(f"Using complete_text instead, length: {len(full_text)}")
                            full_text = str(full_text)
                    
                    # Make sure the text isn't truncated before storage
                    print(f"Final text for DB storage, first 100 chars: {full_text[:100]}")
                    print(f"Final text for DB storage, last 100 chars: {full_text[-100:] if len(full_text) > 100 else full_text}")
                    
                    stored = db_service.store_document(
                        file_path=filepath,
                        analyzed_data=results['ai_analysis']['analyzed_data'],
                        extracted_text=full_text,
                        force_store=True  # Force storage even if other fields are missing
                    )
                    print(f"Document stored successfully: {stored}")
                    
                    try:
                        # Get embeddings for the document
                        print(f"Generating embeddings for document")
                        embedding = db_service._get_embedding(results['full_text'])
                        
                        if embedding:
                            # Find similar documents using vector similarity search
                            print(f"Finding similar documents using vector similarity search")
                            similar_documents = db_service.find_similar_documents(embedding, limit=3)
                            print(f"Found {len(similar_documents)} similar documents")
                            
                            # Add similar documents to the result
                            results['similar_documents'] = similar_documents
                        else:
                            print("Warning: Empty embedding vector generated")
                            results['similar_documents'] = []
                    except Exception as e:
                        print(f"Error during embedding or similarity search: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        results['similar_documents'] = []
                except Exception as e:
                    print(f"Error storing document in database: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
        elif is_pdf and "poppler" in str(results.get('error', '')).lower():
            # Special handling for PDF poppler missing error
            results['is_pdf'] = True
            results['needs_poppler'] = True
        
        return jsonify(results)
    
    return jsonify({'success': False, 'error': 'File type not allowed'})


@app.route('/api/simple', methods=['GET', 'POST'])
def simple_api_process_file():
    # Generate a session ID for WebSocket room if not provided
    session_id = request.args.get('session_id') or str(uuid.uuid4())
    """Simplified API endpoint that automatically handles missing values"""
    # Define fields we're interested in tracking
    fields_to_check = [
        # Original 9 fields
        "Build Year", 
        "Climate Score", 
        "Flood Risk Score", 
        "Infrastructure Opacity", 
        "Land Use Code", 
        "Mls Listing", 
        "Outdated Tax Delta", 
        "Regional Data Variation", 
        "Zoning Record",
        # 5 additional fields
        "Property Address",
        "Owner Name",
        "Parcel Number",
        "Property Type & Use",
        "Legal Description"
    ]

    # Check if this is a field update request
    if request.is_json and request.json and 'session_id' in request.json:
        # Handle field updates
        session_id = request.json['session_id']
        # Emit socket event that processing has started
        send_processing_status(socketio, session_id, 'processing_started', message='Field update received')
        field_updates = request.json.get('field_updates', {})
        
        if not field_updates:
            return jsonify({'error': 'No field updates provided'})
        
        # Retrieve the temporary data
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{session_id}.json")
        if not os.path.exists(temp_file):
            return jsonify({'error': 'Session expired or invalid'})
        
        try:
            with open(temp_file, 'r') as f:
                temp_data = json.load(f)
                
            filepath = temp_data['filepath']
            results = temp_data['results']
            missing_fields = temp_data['missing_fields']
            
            # Update the missing fields
            for field in missing_fields:
                if field in field_updates:
                    if 'ai_analysis' not in results:
                        results['ai_analysis'] = {}
                    if 'analyzed_data' not in results['ai_analysis']:
                        results['ai_analysis']['analyzed_data'] = {}
                    results['ai_analysis']['analyzed_data'][field] = field_updates[field]
                    # Emit socket event for each field update
                    send_field_updated(socketio, session_id, field, field_updates[field])
            
            # Check if all fields have been filled in
            still_missing = []
            for field in fields_to_check:
                if results['ai_analysis']['analyzed_data'].get(field, 'not found') == 'not found':
                    still_missing.append(field)
                    
            # Emit socket event with current state of missing fields
            fields_found = {k: v for k, v in results['ai_analysis']['analyzed_data'].items() 
                          if k in fields_to_check and v != 'not found'}
            send_missing_fields_update(socketio, session_id, fields_found, still_missing)
                    
            # If all fields are complete, finalize processing and return complete result
            if not still_missing:
                # Save updated results to the JSON file in archive
                if 'filename' in results:
                    json_path = os.path.join(app.config['ARCHIVE_FOLDER'], f"{os.path.splitext(results['filename'])[0]}.json")
                    with open(json_path, 'w') as f:
                        json.dump(results, f)
                
                # If we have analyzed data, update the database
                if 'ai_analysis' in results and 'analyzed_data' in results['ai_analysis']:
                    try:
                        # Extract full text
                        full_text = results.get('full_text', '')
                        if not full_text and 'ai_analysis' in results and 'analyzed_data' in results['ai_analysis']:
                            full_text = results['ai_analysis']['analyzed_data'].get('complete_text', '')
                        
                        # Update database
                        db_service.store_document(
                            file_path=filepath,
                            analyzed_data=results['ai_analysis']['analyzed_data'],
                            extracted_text=full_text,
                            force_store=True
                        )
                    except Exception as e:
                        print(f"Error updating database: {str(e)}")
                
                # Create response with all fields
                api_response = {}
                for field in fields_to_check:
                    value = results['ai_analysis']['analyzed_data'].get(field)
                    if value and value != 'not found':
                        api_response[field] = value
                
                # Generate RAG response with similar documents
                try:
                    # Create embedding text from all 14 fields
                    document_data = {}
                    for field in fields_to_check:
                        document_data[field] = results['ai_analysis']['analyzed_data'].get(field, '0')
                    
                    # Create text for embedding from all 14 fields
                    embedding_text = ''
                    for field in fields_to_check:
                        embedding_text += f"{field}: {document_data[field]}\n"
                    
                    # Generate embedding
                    db_service.connect()
                    embedding = db_service._get_embedding(embedding_text)
                    
                    if embedding is not None:
                        # Find similar documents
                        similar_docs = db_service.find_similar_documents(embedding, limit=3)
                        
                        # Calculate risk score for the current document
                        risk_score = db_service._calculate_risk_score(document_data)
                        
                        # Format similar documents according to the screenshot example
                        rag_results = []
                        
                        for doc in similar_docs:
                            # Calculate similarity percentage based on risk score proximity
                            doc_risk_score = int(doc['risk_score'])
                            current_risk = risk_score if risk_score is not None else 0
                            
                            # Calculate similarity based on risk score proximity
                            max_risk_diff = 100  # Maximum possible difference
                            risk_diff = abs(current_risk - doc_risk_score)
                            similarity_percentage = max(0, 100 - (risk_diff * 100 / max_risk_diff))
                            
                            doc_name = doc['file_name']
                            
                            # Format document response
                            doc_response = {
                                "Document Name": doc_name,
                                "Similarity Score": f"{similarity_percentage:.1f}%",
                                "Risk Score": doc_risk_score
                            }
                            rag_results.append(doc_response)
                        
                        # Add RAG results to the API response
                        api_response["rag_response"] = rag_results
                except Exception as e:
                    print(f"Error generating RAG response: {str(e)}")
                    # If RAG generation fails, still return the fields
                
                # Clean up temp file
                os.remove(temp_file)
                
                # Send socket event that processing is complete
                send_process_complete(socketio, session_id, api_response)
                
                # Return combined response with fields and RAG data
                return jsonify(api_response)
            else:
                # Update the temp data with remaining missing fields
                temp_data['missing_fields'] = still_missing
                with open(temp_file, 'w') as f:
                    json.dump(temp_data, f)
                    
                # Return error with instructions for filling missing fields
                return jsonify({
                    'error': 'Missing fields need to be filled',
                    'session_id': session_id,
                    'missing_fields': still_missing
                })
                    
        except Exception as e:
            print(f"Error processing update: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error updating fields: {str(e)}'})
    
    # This is a file upload request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Create a unique filename to avoid collisions
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        new_filename = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        
        # Ensure upload folder exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Process the file based on file type
        file_ext = os.path.splitext(filepath)[1].lower()
        try:
            if file_ext in ['.xlsx', '.csv']:
                # Process Excel or CSV file
                results = process_excel_csv_file(filepath)
            else:
                # Process image/PDF with OCR processor
                ocr_processor = OCRProcessor()
                results = ocr_processor.process_image(filepath)
            
            # Return only the specific fields requested from analyzed_data
            if results.get('success', False) and 'ai_analysis' in results and 'analyzed_data' in results['ai_analysis']:
                # Check for missing fields
                missing_fields = []
                for field in fields_to_check:
                    if results['ai_analysis']['analyzed_data'].get(field, 'not found') == 'not found':
                        missing_fields.append(field)
                
                # If no missing fields, return the data directly
                if not missing_fields:
                    # Create a filtered response with only the fields that have values
                    api_response = {}
                    for field in fields_to_check:
                        value = results['ai_analysis']['analyzed_data'].get(field)
                        if value and value != 'not found':
                            api_response[field] = value
                    
                    # Send socket event that all fields are found and processing is complete
                    send_processing_status(socketio, session_id, 'processing_complete', message='All fields extracted successfully')
                    send_process_complete(socketio, session_id, api_response)
                    return jsonify(api_response)
                else:
                    # We have missing fields - save the current state and provide session info
                    session_id = str(uuid.uuid4())
                    temp_data = {
                        'filepath': filepath,
                        'results': results,
                        'missing_fields': missing_fields
                    }
                    
                    # Send socket event with missing fields information
                    fields_found = {field: results['ai_analysis']['analyzed_data'].get(field) 
                                  for field in fields_to_check 
                                  if field not in missing_fields and results['ai_analysis']['analyzed_data'].get(field) != 'not found'}
                    send_processing_status(socketio, session_id, 'missing_fields_detected', 
                                       message=f'Found {len(fields_found)} fields, missing {len(missing_fields)} fields')
                    send_missing_fields_update(socketio, session_id, fields_found, missing_fields)
                    
                    # Store temp data for later retrieval
                    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{session_id}.json")
                    with open(temp_file, 'w') as f:
                        json.dump(temp_data, f)
                    
                    # Return a structured response indicating missing fields and how to proceed
                    found_fields = {}
                    for field in fields_to_check:
                        value = results['ai_analysis']['analyzed_data'].get(field)
                        if value and value != 'not found':
                            found_fields[field] = value
                    
                    # Create a clear response with instructions
                    api_response = {
                        "success": True,
                        "complete": False,
                        "message": "Missing fields detected. Please provide the required values using the /api/missing_value endpoint.",
                        "session_id": session_id,
                        "missing_fields": missing_fields,
                        "found_fields": found_fields,
                        "next_steps": {
                            "endpoint": "/api/missing_value",
                            "method": "POST",
                            "request_format": {
                                "session_id": session_id,
                                "field_updates": {
                                    "example_field_1": "value1",
                                    "example_field_2": "value2"
                                }
                            }
                        }
                    }
                    
                    # No more terminal prompting - return the API response with instructions
                    return jsonify(api_response)
            else:
                return jsonify({'error': 'Processing successful but no analysis data available'})
                
        except Exception as e:
            print(f"Exception during file processing: {str(e)}")
            import traceback
            traceback.print_exc()
            send_error(socketio, session_id, 'upload_error', f'Error processing file: {str(e)}')
            return jsonify({'error': f'Error processing file: {str(e)}'})
    
    return jsonify({'error': 'File type not allowed'})
    
    # This is a file upload request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Create a unique filename to avoid collisions
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        new_filename = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        
        # Ensure upload folder exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Process the file based on file type
        file_ext = os.path.splitext(filepath)[1].lower()
        try:
            if file_ext in ['.xlsx', '.csv']:
                # Process Excel or CSV file
                results = process_excel_csv_file(filepath)
            else:
                # Process image/PDF with OCR processor
                ocr_processor = OCRProcessor()
                results = ocr_processor.process_image(filepath)
            
            # Return only the specific fields requested from analyzed_data
            if results.get('success', False) and 'ai_analysis' in results and 'analyzed_data' in results['ai_analysis']:
                # Only include fields that don't have 'not found' values
                api_response = {}
                missing_fields = []
                for field in fields_to_check:
                    value = results['ai_analysis']['analyzed_data'].get(field, 'not found')
                    if value != 'not found':
                        api_response[field] = value
                    else:
                        missing_fields.append(field)
                        
                # Store the document ID for later updates
                # Store original file path and document ID in session or temporary storage
                if missing_fields:
                    # Save the filepath and results for later retrieval
                    session_id = str(uuid.uuid4())
                    temp_data = {
                        'filepath': filepath,
                        'results': results,
                        'missing_fields': missing_fields
                    }
                    
                    # In production you would use a proper session store
                    # For this example we'll use a file-based store
                    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{session_id}.json")
                    with open(temp_file, 'w') as f:
                        json.dump(temp_data, f)
                    
                    # Add session ID to response for client to use when submitting missing fields
                    api_response['_session_id'] = session_id
                    api_response['_missing_fields'] = missing_fields
                
                return jsonify(api_response)
            else:
                return jsonify({'success': False, 'error': 'Processing successful but no analysis data available'})
                
        except Exception as e:
            print(f"Exception during file processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'})
    
    return jsonify({'success': False, 'error': 'File type not allowed'})



@app.route('/api/update-missing', methods=['POST'])
def update_missing_fields():
    """Endpoint to update missing fields provided by user"""
    # Implementation remains the same
    
@app.route('/api/missing_value', methods=['POST'])
def missing_value():
    """Interactive endpoint for handling missing field values with Postman"""
    data = request.json
    if not data or 'session_id' not in data:
        return jsonify({'error': 'No session ID provided'})
    
    session_id = data['session_id']
    field_updates = data.get('field_updates', {})
    
    if not field_updates:
        return jsonify({'error': 'No field updates provided'})
    
    # Retrieve the temporary data
    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{session_id}.json")
    if not os.path.exists(temp_file):
        return jsonify({'error': 'Session expired or invalid'})
    
    try:
        with open(temp_file, 'r') as f:
            temp_data = json.load(f)
            
        filepath = temp_data['filepath']
        results = temp_data['results']
        missing_fields = temp_data['missing_fields']
        
        # Update the missing fields
        for field in missing_fields:
            if field in field_updates:
                if 'ai_analysis' not in results:
                    results['ai_analysis'] = {}
                if 'analyzed_data' not in results['ai_analysis']:
                    results['ai_analysis']['analyzed_data'] = {}
                results['ai_analysis']['analyzed_data'][field] = field_updates[field]
        
        # Check if all fields have been filled in
        fields_to_check = [
            # Original 9 fields
            "Build Year", 
            "Climate Score", 
            "Flood Risk Score", 
            "Infrastructure Opacity", 
            "Land Use Code", 
            "Mls Listing", 
            "Outdated Tax Delta", 
            "Regional Data Variation", 
            "Zoning Record",
            # 5 additional fields
            "Property Address",
            "Owner Name",
            "Parcel Number",
            "Property Type & Use",
            "Legal Description"
        ]
        
        still_missing = []
        for field in fields_to_check:
            if results['ai_analysis']['analyzed_data'].get(field, 'not found') == 'not found':
                still_missing.append(field)
        
        # Update the temp data with potentially reduced list of missing fields
        temp_data['missing_fields'] = still_missing
        with open(temp_file, 'w') as f:
            json.dump(temp_data, f)
        
        # If all fields are complete, process and return complete result
        if not still_missing:
            # Save updated results to the JSON file in archive
            if 'filename' in results:
                json_path = os.path.join(app.config['ARCHIVE_FOLDER'], f"{os.path.splitext(results['filename'])[0]}.json")
                with open(json_path, 'w') as f:
                    json.dump(results, f)
            
            # If we have analyzed data, update the database
            if 'ai_analysis' in results and 'analyzed_data' in results['ai_analysis']:
                try:
                    # Extract full text
                    full_text = results.get('full_text', '')
                    if not full_text and 'ai_analysis' in results and 'analyzed_data' in results['ai_analysis']:
                        full_text = results['ai_analysis']['analyzed_data'].get('complete_text', '')
                    
                    # Update database
                    db_service.store_document(
                        file_path=filepath,
                        analyzed_data=results['ai_analysis']['analyzed_data'],
                        extracted_text=full_text,
                        force_store=True
                    )
                except Exception as e:
                    print(f"Error updating database: {str(e)}")
            
            # Generate RAG response with similar documents
            try:
                # Create embedding text from all fields
                document_data = {}
                for field in fields_to_check:
                    document_data[field] = results['ai_analysis']['analyzed_data'].get(field, '0')
                
                # Create text for embedding
                embedding_text = ''
                for field in fields_to_check:
                    embedding_text += f"{field}: {document_data[field]}\n"
                
                # Generate embedding
                db_service.connect()
                embedding = db_service._get_embedding(embedding_text)
                
                if embedding is not None:
                    # Find similar documents
                    similar_docs = db_service.find_similar_documents(embedding, limit=3)
                    
                    # Calculate risk score for the current document
                    risk_score = db_service._calculate_risk_score(document_data)
                    
                    # Format similar documents
                    rag_results = []
                    
                    for doc in similar_docs:
                        # Calculate similarity percentage based on risk score proximity
                        doc_risk_score = int(doc['risk_score'])
                        current_risk = risk_score if risk_score is not None else 0
                        
                        # Calculate similarity based on risk score proximity
                        max_risk_diff = 100  # Maximum possible difference
                        risk_diff = abs(current_risk - doc_risk_score)
                        similarity_percentage = max(0, 100 - (risk_diff * 100 / max_risk_diff))
                        
                        doc_name = doc['file_name']
                        
                        # Format document response
                        doc_response = {
                            "Document Name": doc_name,
                            "Similarity Score": f"{similarity_percentage:.1f}%",
                            "Risk Score": doc_risk_score
                        }
                        rag_results.append(doc_response)
                    
                    # Add RAG results to the API response
                    api_response = {}
                    for field in fields_to_check:
                        value = results['ai_analysis']['analyzed_data'].get(field)
                        if value and value != 'not found':
                            api_response[field] = value
                    
                    api_response["rag_response"] = rag_results
                    
                    # Clean up temp file
                    os.remove(temp_file)
                    
                    return jsonify({
                        "success": True,
                        "complete": True,
                        "data": api_response
                    })
            except Exception as e:
                print(f"Error generating RAG response: {str(e)}")
        
        # Return status with remaining missing fields
        return jsonify({
            "success": True,
            "complete": False,
            "session_id": session_id,
            "missing_fields": still_missing
        })
        
    except Exception as e:
        print(f"Error processing missing values: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error updating fields: {str(e)}'})

    
    session_id = data['session_id']
    field_updates = data.get('field_updates', {})
    
    if not field_updates:
        return jsonify({'success': False, 'error': 'No field updates provided'})
    
    # Retrieve the temporary data
    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{session_id}.json")
    if not os.path.exists(temp_file):
        return jsonify({'success': False, 'error': 'Session expired or invalid'})
    
    try:
        with open(temp_file, 'r') as f:
            temp_data = json.load(f)
            
        filepath = temp_data['filepath']
        results = temp_data['results']
        missing_fields = temp_data['missing_fields']
        
        # Update the missing fields
        for field in missing_fields:
            if field in field_updates:
                if 'ai_analysis' not in results:
                    results['ai_analysis'] = {}
                if 'analyzed_data' not in results['ai_analysis']:
                    results['ai_analysis']['analyzed_data'] = {}
                results['ai_analysis']['analyzed_data'][field] = field_updates[field]
        
        # Save updated results to the JSON file in archive
        if 'filename' in results:
            json_path = os.path.join(app.config['ARCHIVE_FOLDER'], f"{os.path.splitext(results['filename'])[0]}.json")
            with open(json_path, 'w') as f:
                json.dump(results, f)
        
        # If we have analyzed data, update the database
        if 'ai_analysis' in results and 'analyzed_data' in results['ai_analysis']:
            try:
                # Extract full text
                full_text = results.get('full_text', '')
                if not full_text and 'ai_analysis' in results and 'analyzed_data' in results['ai_analysis']:
                    full_text = results['ai_analysis']['analyzed_data'].get('complete_text', '')
                
                # Update database
                db_service.store_document(
                    file_path=filepath,
                    analyzed_data=results['ai_analysis']['analyzed_data'],
                    extracted_text=full_text,
                    force_store=True
                )
            except Exception as e:
                print(f"Error updating database: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create response with all fields (including updated ones)
        fields_to_include = [
            # Original 9 fields
            "Build Year", 
            "Climate Score", 
            "Flood Risk Score", 
            "Infrastructure Opacity", 
            "Land Use Code", 
            "Mls Listing", 
            "Outdated Tax Delta", 
            "Regional Data Variation", 
            "Zoning Record",
            # 5 additional fields
            "Property Address",
            "Owner Name",
            "Parcel Number",
            "Property Type & Use",
            "Legal Description"
        ]
        
        api_response = {}
        for field in fields_to_include:
            value = results['ai_analysis']['analyzed_data'].get(field, 'not found')
            if value != 'not found':
                api_response[field] = value
        
        # Clean up temp file
        os.remove(temp_file)
        
        return jsonify(api_response)
                
    except Exception as e:
        print(f"Error processing update: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error updating fields: {str(e)}'})



@app.route('/api/rag_response', methods=['POST'])
def rag_response():
    """RAG response API that calculates document similarity and risk scores
    
    Returns similar documents based on embedding similarity and calculates risk scores
    using the YAML adopter based on 9 required fields.
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract the 14 required fields from the request
        required_fields = [
            # Original 9 fields
            "Build Year", 
            "Climate Score", 
            "Flood Risk Score", 
            "Infrastructure Opacity", 
            "Land Use Code", 
            "Mls Listing", 
            "Outdated Tax Delta", 
            "Regional Data Variation", 
            "Zoning Record",
            # 5 additional fields
            "Property Address",
            "Owner Name",
            "Parcel Number",
            "Property Type & Use",
            "Legal Description"
        ]
        
        # Check if all required fields are present
        document_data = {}
        missing_fields = []
        for field in required_fields:
            if field in data and data[field] is not None and data[field] != '':
                document_data[field] = data[field]
            else:
                missing_fields.append(field)
                document_data[field] = "not found"  # Default value for missing fields
        
        # Fill missing values if any
        if missing_fields:
            # Here you would typically use some method to fill missing values
            # For now, we'll just use a placeholder value
            for field in missing_fields:
                document_data[field] = "0"  # Default placeholder
        
        # Create text for embedding from the 9 fields
        embedding_text = ''
        for field in required_fields:
            embedding_text += f"{field}: {document_data[field]}\n"
        
        # Generate embedding using text embedding small 03
        db_service.connect()
        embedding = db_service._get_embedding(embedding_text)
        
        # If embedding generation failed, return error
        if embedding is None:
            return jsonify({'error': 'Failed to generate embedding for document'}), 500
        
        # Find similar documents
        similar_docs = db_service.find_similar_documents(embedding, limit=3)
        
        # Calculate risk score for the current document
        risk_score = db_service._calculate_risk_score(document_data)
        
        # Format similar documents according to the screenshot example
        # We're returning a top-level array of similar documents to match the screenshot
        response = []
        
        # Add the current document as an implicit entry if needed
        # Skip this if you don't want the current document in the response
        
        # Add the similar documents to the response
        for doc in similar_docs:
            similarity_percentage = float(doc['similarity']) * 100
            doc_name = doc['file_name']
            
            # Format to exactly match the screenshot pattern
            # Using the exact same field names and structure
            doc_response = {
                "Document Name": doc_name,
                "Similarity Score": f"{similarity_percentage:.1f}%",
                "Risk Score": int(doc['risk_score'])
            }
            response.append(doc_response)
        
        # Return the RAG response exactly as shown in the screenshot
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in RAG response API: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing RAG response: {str(e)}'}), 500

@app.route('/api')
def api_home():
    return jsonify({"status": "API running", "endpoints": ["/api/simple", "/api/missing_value"]})

if __name__ == '__main__':
    # Use eventlet with SocketIO instead of regular app.run
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
