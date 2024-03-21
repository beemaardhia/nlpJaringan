import pandas as pd
import streamlit as st
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pickle


stopwords_ind = stopwords.words('indonesian')

# Buat fungsi untuk langkah case folding
def casefolding(text):
    text = text.lower()   # Mengubah teks menjadi lower case    
    text = re.sub(r'#\w+\s*', '', text)
    text = re.sub(r'<[^>]+>|\n', ' ', text)            # Menghapus <USERNAME>                      
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Menghapus URL
    text = re.sub(r'[-+]?[0-9]+', '', text)           # Menghapus angka
    text = re.sub(r'[^\w\s]', '', text)               # Menghapus karakter tanda baca
    text = re.sub(r'\busername\b', 'username', text, flags=re.IGNORECASE)  # Mengganti 'username' dengan 'username' (case-insensitive)
    return text

key_norm = pd.read_csv('key_norm_indo.csv')

def text_normalize(text):
  text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in text.split()])
  text = str.lower(text)
  return text

def remove_stop_words(text):
  clean_words = []
  text = text.split()
  for word in text:
      if word not in stopwords_ind:
          clean_words.append(word)
  return " ".join(clean_words)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Buat fungsi untuk langkah stemming bahasa Indonesia
def stemming(text):
  text = stemmer.stem(text)
  return text

def text_preprocessing_process(text):
  text = casefolding(text)
  text = text_normalize(text)
  text = remove_stop_words(text)
  text = stemming(text)
  return text

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Mengimpor model dari file .joblib
model = joblib.load('nlpJarinan.joblib')
vocab = pickle.load(open('kbest_feature.pickle', 'rb'))

def predict_sentiment(text):
    preprocessed_text = text_preprocessing_process(text)
    tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))  # Sesuaikan dengan vektorizer yang Anda gunakan
    result = model.predict(tf_idf_vec.fit_transform([preprocessed_text]))
    return result


# Tampilan Streamlit
st.header('Sentiment Analysis')

with st.expander('Analyze Text'):
    input_text = st.text_area('Text here:', '')  # Menerima input teks dari pengguna

    if st.button('Analyze'):  # Tombol untuk melakukan analisis
        if input_text:
            prediction = predict_sentiment(input_text)
            st.write('Hasil Prediksi:', prediction)
