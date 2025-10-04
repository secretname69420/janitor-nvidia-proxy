from flask import Flask, request, jsonify, Response
import requests
import os
import json
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": "*"}})

# NVIDIA NIM API configuration
NIM_API_KEY = os.environ.get('NIM_API_KEY', 'not-configured')
NIM_BASE_URL = 'https://integrate.api.nvidia.com/v1'

# Log all incoming requests
@app.before_request
def log_request_info():
    logger.info('='*50)
    logger.info(f'REQUEST: {request.method} {request.path}')
    logger.info(f'From: {request.remote_addr}')
    logger.info('='*50)

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
    
    try:
        data = request.json
        logger.info(f"Received chat request with model: {data.get('model', 'not specified')}")
        
        # Validate API key
        if NIM_API_KEY == 'not-configured':
            logger.error("❌ API key not configured!")
            return jsonify({'error': 'API key not configured on server. Please set NIM_API_KEY environment variable.'}), 500
        
        # Map OpenAI format to NVIDIA NIM format
        nim_request = {
            'model': data.get('model', 'meta/llama-3.3-70b-instruct'),
            'messages': data.get('messages', []),
            'temperature': data.get('temperature', 0.7),
            'top_p': data.get('top_p', 1.0),
            'max_tokens': data.get('max_tokens', 1024),
            'stream': data.get('stream', False)
        }
        
        headers = {
            'Authorization': f'Bearer {NIM_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"Forwarding to NVIDIA API...")
        
        # Handle streaming
        if nim_request['stream']:
            nim_response = requests.post(
                f'{NIM_BASE_URL}/chat/completions',
                headers=headers,
                json=nim_request,
                stream=True,
                timeout=120
            )
            
            logger.info(f"NVIDIA response status: {nim_response.status_code}")
            
            if nim_response.status_code != 200:
                error_text = nim_response.text
                logger.error(f"❌ NVIDIA API error: {error_text}")
                return jsonify({'error': error_text}), nim_response.status_code
            
            def generate():
                for chunk in nim_response.iter_lines():
                    if chunk:
                        yield chunk + b'\n'
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            nim_response = requests.post(
                f'{NIM_BASE_URL}/chat/completions',
                headers=headers,
                json=nim_request,
                timeout=120
            )
            
            logger.info(f"✅ NVIDIA response status: {nim_response.status_code}")
            
            if nim_response.status_code != 200:
                logger.error(f"❌ NVIDIA API error: {nim_response.text}")
                return jsonify({
                    'error': 'NVIDIA API Error',
                    'details': nim_response.text,
                    'status_code': nim_response.status_code
                }), nim_response.status_code
            
            return jsonify(nim_response.json()), nim_response.status_code
            
    except requests.exceptions.Timeout:
        logger.error("❌ Request to NVIDIA API timed out")
        return jsonify({'error': 'Request timed out'}), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request error: {str(e)}")
        return jsonify({'error': f'Request error: {str(e)}'}), 502
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    logger.info("Models list requested")
    return jsonify({
        'object': 'list',
        'data': [
            {
                'id': 'meta/llama-3.3-70b-instruct',
                'object': 'model',
                'created': 1686935002,
                'owned_by': 'nvidia'
            }
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    api_configured = NIM_API_KEY != 'not-configured' and len(NIM_API_KEY) > 10
    logger.info(f"Health check - API configured: {api_configured}")
    return jsonify({
        'status': 'healthy',
        'api_key_configured': api_configured,
        'base_url': NIM_BASE_URL,
        'model': 'meta/llama-3.3-70b-instruct'
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': '🚀 NVIDIA Llama 3.3 Proxy Running',
        'model': 'meta/llama-3.3-70b-instruct',
        'endpoints': {
            '/v1/chat/completions': 'POST - Chat completions',
            '/v1/models': 'GET - List models',
            '/health': 'GET - Health check'
        },
        'usage': 'Point Janitor AI to this URL + /v1'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on port {port}")
    logger.info(f"API Key configured: {NIM_API_KEY != 'not-configured'}")
    app.run(host='0.0.0.0', port=port)
