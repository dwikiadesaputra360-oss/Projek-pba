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

data = pd.read_csv('Wisata_fixed.csv')

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

import difflib

# =====================================================
# KAMUS SINONIM & TYPO CORRECTION
# =====================================================

synonyms = {
    'ongkos': 'harga',
    'biaya': 'harga',
    'tarif': 'harga',
    'karcis': 'tiket',
    'karcisnya': 'tiket',
    'tempat': 'lokasi',
    'letak': 'lokasi',
    'posisi': 'lokasi',
    'tujuan': 'destinasi',
    'makanan': 'makan',
    'minuman': 'minum'
}

def correct_input(text, vocab):
    words = text.split()
    corrected = []
    for w in words:
        if w in synonyms:
            w = synonyms[w]
        
        if w in vocab:
            corrected.append(w)
        else:
            # Toleransi Typo (Spell Checker)
            matches = difflib.get_close_matches(w, vocab, n=1, cutoff=0.7)
            if matches:
                corrected.append(matches[0])
            else:
                corrected.append(w)
    return ' '.join(corrected)

# =====================================================
# SIAPKAN MODEL CHATBOT
# =====================================================

data['pertanyaan_preprocessed'] = data['pertanyaan'].apply(preprocessing)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['pertanyaan_preprocessed'])
vocab_set = set(vectorizer.vocabulary_.keys())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    # 1. Preprocessing dasar
    user_message_clean = preprocessing(user_message)

    if not user_message_clean.strip():
        return jsonify({'response': 'Silakan tulis pertanyaan terlebih dahulu.'})

    # 2. Toleransi Typo & Sinonim
    user_message_corrected = correct_input(user_message_clean, vocab_set)

    user_words = user_message_corrected.split()
    known_words = [w for w in user_words if w in vocab_set]

    # Validasi tambahan
    if len(known_words) / len(user_words) < 0.5:
        return jsonify({'response': 'Maaf, pertanyaan tersebut berada di luar pengetahuan saya. Silakan tanyakan hal lain seputar wisata ini.'})

    # 3. TF-IDF Cosine Similarity
    user_vec = vectorizer.transform([user_message_corrected])
    scores = cosine_similarity(user_vec, tfidf_matrix)[0]
    best_idx = scores.argmax()
    best_score = scores[best_idx]

    if best_score < 0.75:
        response = 'Maaf, pertanyaan tersebut berada di luar pengetahuan saya. Silakan tanyakan hal lain seputar wisata ini.'
    else:
        response = data.loc[best_idx, 'jawaban']
        
        # 4. Tampilan Balasan Multimedia (Google Maps)
        # Jika jawaban mengandung kata kunci lokasi, tambahkan iframe Peta
        pertanyaan_cocok = data.loc[best_idx, 'pertanyaan'].lower()
        if 'lokasi' in pertanyaan_cocok or 'dimana' in pertanyaan_cocok:
            # TOLONG JANGAN DIGANTI LINK NYA KE GOOGLE MAPS BIASA.
            # GOOGLE AKAN MEMBLOKIR PETA JIKA MENGGUNAKAN LINK BIASA DI DALAM WEBSITE.
            # SAYA SUDAH MENGAMBIL LINK DARI ANDA DAN MENGUBAHNYA KE FORMAT "EMBED" (SEMATKAN).
            map_html = '<br><iframe src="https://maps.google.com/maps?q=-5.4178495,105.1866787&hl=id-ID&z=14&output=embed" width="100%" height="200" style="border:0; border-radius: 8px; margin-top: 15px;" allowfullscreen="" loading="lazy"></iframe>'
            response += map_html

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')