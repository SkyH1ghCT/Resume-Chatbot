from flask import Flask, request, render_template, jsonify
import fitz  # PyMuPDF
import re
import string
from transformers import pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# 📥 Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# 🧹 Clean the extracted text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text

# Sample training data for prediction
data = {
    'Resume': [
        "Experienced software engineer with expertise in Python, Java, and web development.",
        "Data scientist with experience in machine learning, data analysis, and Python.",
        "Project manager with strong leadership skills and experience managing software projects.",
        "Junior developer with knowledge of Python, HTML, and basic SQL.",
        "HR specialist with strong communication skills and knowledge of recruitment processes."
    ],
    'Job Role': ['Software Engineer', 'Data Scientist', 'Project Manager', 'Junior Developer', 'HR Specialist']
}
df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df['Resume'], df['Job Role'], test_size=0.3, random_state=42)

# Train the model
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return 'No file uploaded', 400
    resume_file = request.files['resume']
    if resume_file.filename == '':
        return 'No file selected', 400
    text = extract_text_from_pdf(resume_file)
    cleaned_text = clean_text(text)
    resume_tfidf = vectorizer.transform([cleaned_text])
    predicted_role = model.predict(resume_tfidf)[0]
    
    return jsonify({'predicted_role': predicted_role})

if __name__ == '__main__':
    app.run(debug=True)
