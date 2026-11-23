from flask import Flask, request, url_for,render_template, jsonify, redirect
import os
import PyPDF2
from werkzeug.utils import secure_filename
from gensim.models import Doc2Vec, FastText
from gensim.models.doc2vec import TaggedDocument 
import pickle
import joblib
import re
import numpy as np
import nltk
import pandas as pd
nltk.download('punkt_tab')
from flask_cors import CORS


app=Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#----------------------------
#       Helper
#----------------------------

def extract_text_from_pdf(file):
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    
    return text
def preprocess_text(text):
        "preprocess text"
        # Convert the text to lowercase
        text = text.lower()
        # Remove punctuation from the text
        text = re.sub('[^a-z]', ' ', text)
        # Remove numerical values from the text
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespaces
        text = ' '.join(text.split())
        return text

def calculate_similarity(text1, text2):
    "get the similarity  score"
    model = Doc2Vec.load('doc2vec.pkl')
    v1 = model.infer_vector(text1.split())
    v2 = model.infer_vector(text2.split())
    similarity = 100*(np.dot(np.array(v1), np.array(v2))) / (np.linalg.norm(np.array(v1)) * np.linalg.norm(np.array(v2)))    
    return similarity 

     
#----------------------------
#       App
#----------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/job-title-data',methods=['GET'])
def get_job_title_data():
   
        # Load the dataset
        df = pd.read_csv('train.csv')
        
        # Get value counts for job titles
        job_counts = df['Agency'].value_counts().head(20)  # Top 20 most common
        posting_type=df['Posting Type'].value_counts()
        
        # Prepare data for chart
        chart_data = {
            'labels': job_counts.index.tolist(),
            'counts': job_counts.values.tolist(),
            'posting_type_labels': posting_type.index.tolist(),
            'posting_type_counts':posting_type.values.tolist()
        }
        
        return jsonify(chart_data)
    
 

     

@app.route('/compare', methods=['POST'])
def compare_files():    
    # Get both files
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')
    
    if not file1 or not file2:
        return jsonify({'error': 'Please upload both files'}), 400
        
    # Save files temporarily
    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
    
    file1.save(file1_path)
    file2.save(file2_path)
    try:
        text1 = extract_text_from_pdf(file1_path)
        text2 = extract_text_from_pdf(file2_path)

        # Scanned/image-only PDFs have no extractable text. Previously this
        # path fell through and returned None, which Flask turns into an
        # opaque 500 error.
        if not text1.strip() or not text2.strip():
            return jsonify({
                'error': 'Could not read any text from one of the PDFs. '
                         'Scanned or image-only PDFs are not supported.',
                'status': 'error'
            }), 400

        score = calculate_similarity(text1, text2)
        if score is None:
            raise ValueError("Similarity calculation returned None")

        return jsonify({
            'score': round(float(score), 2)  # Round to 2 decimal places
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
    finally:
        # Always remove the uploaded files, including on errors.
        for path in (file1_path, file2_path):
            if os.path.exists(path):
                os.remove(path)

if __name__ == '__main__':
    app.run(debug=True)