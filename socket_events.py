"""
Socket event handling module for real-time progress updates
This module contains helper functions for sending events to connected clients
"""

def send_upload_progress(socketio, session_id, progress):
    """
    Send upload progress updates to clients
    
    Args:
        socketio: The SocketIO instance
        session_id: The unique session identifier
        progress: Progress percentage (0-100) or status object
    """
    data = {
        'event': 'upload_progress',
        'session_id': session_id,
        'progress': progress
    }
    socketio.emit('progress_update', data, room=session_id)

def send_processing_status(socketio, session_id, status, step=None, message=None):
    """
    Send processing status updates to clients
    
    Args:
        socketio: The SocketIO instance
        session_id: The unique session identifier
        status: Status string (e.g., 'extracting_fields', 'generating_embeddings')
        step: Optional step number or identifier
        message: Optional status message
    """
    data = {
        'event': 'processing_status',
        'session_id': session_id,
        'status': status
    }
    
    if step is not None:
        data['step'] = step
    
    if message is not None:
        data['message'] = message
        
    socketio.emit('progress_update', data, room=session_id)

def send_missing_fields_update(socketio, session_id, fields_found, missing_fields):
    """
    Send update about missing fields
    
    Args:
        socketio: The SocketIO instance
        session_id: The unique session identifier
        fields_found: Dictionary of fields successfully extracted
        missing_fields: List of fields that are missing
    """
    data = {
        'event': 'missing_fields',
        'session_id': session_id,
        'fields_found': fields_found,
        'missing_fields': missing_fields
    }
    socketio.emit('progress_update', data, room=session_id)

def send_field_updated(socketio, session_id, field, value):
    """
    Send notification when a field is updated
    
    Args:
        socketio: The SocketIO instance
        session_id: The unique session identifier
        field: The field name that was updated
        value: The new value of the field
    """
    data = {
        'event': 'field_updated',
        'session_id': session_id,
        'field': field,
        'value': value
    }
    socketio.emit('progress_update', data, room=session_id)

def send_process_complete(socketio, session_id, result):
    """
    Send notification when the entire process is complete
    
    Args:
        socketio: The SocketIO instance
        session_id: The unique session identifier
        result: The final result object
    """
    data = {
        'event': 'process_complete',
        'session_id': session_id,
        'result': result
    }
    socketio.emit('progress_update', data, room=session_id)

def send_error(socketio, session_id, error_type, message):
    """
    Send error notification
    
    Args:
        socketio: The SocketIO instance
        session_id: The unique session identifier
        error_type: Type of error
        message: Error message
    """
    data = {
        'event': 'error',
        'session_id': session_id,
        'error_type': error_type,
        'message': message
    }
    socketio.emit('progress_update', data, room=session_id)
