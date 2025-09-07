import os
import logging
from flask import Flask, request, jsonify
import pipeline
from PIL import Image # For handling image data
import numpy as np    # For converting image to array format for your pipeline
import io             # For handling in-memory byte streams

# --- Configuration ---
# No longer need UPLOAD_FOLDER as we process in memory
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Pre-load Models (This part is unchanged) ---
logging.info("Initializing OCR models... This may take a moment.")
recognizer = pipeline.AdvancedTextRecognizer()
recognizer.initialize_models()
extractor = pipeline.InformationExtractor()
preprocessor = pipeline.DrivingLicensePreprocessor()
logging.info("✅ Models initialized and ready to receive requests.")

@app.route('/extract-dl', methods=['POST'])
def extract_dl_info():
    if 'dl_image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400
    
    file = request.files['dl_image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    app.logger.info(f"Processing uploaded file: {file.filename}")

    try:
        # 1. Read the image file stream into memory
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 2. Convert the PIL image to a NumPy array, which your pipeline likely expects
        image_np = np.array(pil_image)

        # 3. Feed the in-memory image directly into your pipeline's preprocessor
        #    This bypasses the need for DrivingLicenseImageInput and saving files.
        processed_image, _ = preprocessor.preprocess_image(image_np, show_steps=False)
        ocr_results = recognizer.recognize_text(processed_image)
        extracted_info = extractor.extract_all_info(ocr_results)
        
        app.logger.info(f"Successfully extracted DL numbers: {extracted_info.dl_numbers}")
        return jsonify({
            'dl_numbers': extracted_info.dl_numbers,
            'name': extracted_info.name,
            'raw_text': extracted_info.raw_text
        })
    except Exception as e:
        app.logger.error(f"Error processing file: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred during image processing: {str(e)}'}), 500

if __name__ == "__main__":
    # This section is not used when deploying with Gunicorn, but is useful for local testing.
    port = int(os.environ.get('PORT', 8080))
    # For local test: app.run(debug=False, host='0.0.0.0', port=port)
    pass