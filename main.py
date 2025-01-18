import streamlit as st
import pandas as pd


starcomp_df = pd.read_csv("beritabasket.csv")
#st.dataframe(starcomp_df)


#st.dataframe(starcomp_df.isnull().sum())
starcomp_df = starcomp_df[starcomp_df['judul berita'].notnull()]

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import re
import random

clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
sastrawi = StopWordRemoverFactory()
stopworda = sastrawi.get_stop_words()
factory=StemmerFactory()
Stemmer = factory.create_stemmer()

def clean_text(text):
  text= text.lower()
  text = clean_spcl.sub(' ', text)
  text = clean_symbol.sub(' ', text)
  text = ' '.join(word for word in text.split() if word not in stopworda)
  return text

starcomp_df['desc_clean'] = starcomp_df['judul berita'].apply(clean_text)


#st.dataframe(starcomp_df)

#print(starcomp_df.columns)
starcomp_df.set_index('judul berita', inplace=True)
tf = TfidfVectorizer(analyzer = 'word', ngram_range=(1, 3), min_df=0.0)
tfidf_matrix = tf.fit_transform(starcomp_df['desc_clean'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#cos_sim
#indices = starcomp_df.index
indices = pd.Series(starcomp_df.index)
#indices[:15]

def recommendations (title, top = 10):

   recommended_novell = []

   matching_indices = indices[indices.str.contains(title, case=False, na=False)]
   idx = matching_indices.index[0]
   score_series = pd.Series (cos_sim[idx]).sort_values (ascending = False)

   top = top + 1
   top_indexes = list (score_series.iloc[0:top].index)

   for i in top_indexes:
       recommended_novell.append(list (starcomp_df.index) [i]+" - "+str(score_series[i]))

   return recommended_novell

st.title("Sistem Rekomendasi berita olahraga")
novel = st.text_input("Masukkan Nama berita olahraga")
rekomendasi = st.button("Rekomendasi")

if rekomendasi:
    st.dataframe(recommendations(novel, 15))

