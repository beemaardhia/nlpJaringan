
import nltk

# Membuat fungsi untuk memeriksa apakah koleksi data sudah terinstal
def check_and_download(collection_name):
    try:
        nltk.data.find(collection_name)
    except LookupError:
        print(f"{collection_name} belum terinstal. Mengunduh koleksi data...")
        nltk.download(collection_name)
        print(f"{collection_name} berhasil diunduh.")
    else:
        print(f"{collection_name} sudah terinstal.")

# Memeriksa dan mengunduh koleksi data 'punkt'
check_and_download('punkt')

# Memeriksa dan mengunduh koleksi data 'stopwords'
check_and_download('stopwords')


import pandas as pd
import streamlit as st
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


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

key_norm = pd.read_csv('lemot/key_norm_indo.csv')

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

# Mengimpor model dari file .joblib
model = joblib.load('lemot/nlpJarinan.joblib')
vocab = pickle.load(open('lemot/kbest_feature.pickle', 'rb'))

def predict_sentiment(text):
    preprocessed_text = text_preprocessing_process(text)
    tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))  # Sesuaikan dengan vektorizer yang Anda gunakan
    result = model.predict(tf_idf_vec.fit_transform([preprocessed_text]))
    return result

def convert_encoding_to_utf8(upl):
    content = upl.read().decode('ANSI', errors='ignore')
    content_utf8 = content.encode('utf-8')
    upl.seek(0)
    upl.write(content_utf8)
    upl.seek(0)


# Tampilan Streamlit
st.header('Sentiment Analysis Komentar Provider')

# Watermark
st.markdown('<p style="color:gray; font-size:small;text-align:center;">created by bimaardhiavardhan</p>', unsafe_allow_html=True)

with st.expander('Analyze Text'):
    input_text = st.text_area('Ketik Komentar Provider:', '')  # Menerima input teks dari pengguna

    if st.button('Analyze'):  # Tombol untuk melakukan analisis
        if input_text:
            prediction = predict_sentiment(input_text)
            st.write('Hasil Prediksi:', prediction)

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')
    

    if upl:
        
        convert_encoding_to_utf8(upl)
        
        df = pd.read_csv(upl, on_bad_lines='skip')
        # del df['Unnamed: 0']

        # Preprocessing pada teks di kolom 'tweets'
        df['preprocessed_text'] = df['Text Tweet'].apply(text_preprocessing_process)


        # Terapkan fungsi analisis sentimen pada setiap baris dalam DataFrame
        df['sentiment_analysis'] = df['preprocessed_text'].apply(predict_sentiment)

        st.write(df[df['sentiment_analysis'] == 'positive'].head())
        st.write(df[df['sentiment_analysis'] == 'negative'].head())

        # Mendownload hasil analisis sebagai file CSV
        @st.cache_data
        def convert_df(df):
            # Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment_analysis.csv',
            mime='text/csv',
        )
