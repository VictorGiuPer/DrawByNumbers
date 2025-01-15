from flask import Flask, render_template, request, redirect, url_for
import sys
import os
# print("Current working directory:", os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from main import start_application
app = Flask(__name__)

UPLOAD_FOLDER = 'ui/data/uploads/'  # Point to the 'data' folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    """Rendering the home page (index.html)"""
    return render_template('index.html')





""" @app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # Save the file to the server
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    try:
        # Call your image processing function from src
        processed_image, processed_png = start_application(filename)
        # Save the processed image or return a response
        processed_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
        processed_png.save(processed_filename)

        # Render a response with the processed image
        return render_template('result.html', filename=processed_filename)
    finally:
        # Clean up: Delete the uploaded file
        if os.path.exists(filename):
            os.remove(filename)
"""

if __name__ == '__main__':
    app.run(debug=True)
