from flask import Flask, render_template, request, send_file, jsonify
import os
import processor
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_key")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save uploaded file temporarily
        temp_path = 'temp_upload.jpg'
        file.save(temp_path)
        
        # Process image using existing code
        results = processor.process_image(temp_path)
        
        # Return results
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup
        if os.path.exists('temp_upload.jpg'):
            os.remove('temp_upload.jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
