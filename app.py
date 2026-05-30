# =====================================================
# IMPORT LIBRARY
# =====================================================

from flask import Flask, render_template, request, jsonify

import pandas as pd
import nltk
import re
import string

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# DOWNLOAD RESOURCE NLTK
# =====================================================

nltk.download('punkt')
nltk.download('stopwords')

# =====================================================
# MEMBUAT FLASK APP
# =====================================================

app = Flask(__name__, static_folder='static', template_folder='templates')

# =====================================================
# MEMBACA DATASET CSV
# =====================================================

data = pd.read_csv('Wisata.csv')

# =====================================================
# STOPWORD INDONESIA
# =====================================================

stop_words = set(stopwords.words('indonesian'))

# =====================================================
# FUNGSI PREPROCESSING
# =====================================================

def preprocessing(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    clean_text = ' '.join(filtered_tokens)
    return clean_text

# =====================================================
# SIAPKAN MODEL CHATBOT
# =====================================================

data['pertanyaan_preprocessed'] = data['pertanyaan'].apply(preprocessing)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['pertanyaan_preprocessed'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    user_message_clean = preprocessing(user_message)

    if not user_message_clean.strip():
        return jsonify({'response': 'Silakan tulis pertanyaan terlebih dahulu.'})

    user_vec = vectorizer.transform([user_message_clean])
    scores = cosine_similarity(user_vec, tfidf_matrix)[0]
    best_idx = scores.argmax()
    best_score = scores[best_idx]

    if best_score < 0.1:
        response = 'Maaf, saya tidak mengerti. Silakan tulis pertanyaan lain.'
    else:
        response = data.loc[best_idx, 'jawaban']

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')