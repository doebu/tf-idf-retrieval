import os
import re
import math
import pandas as pd
import nltk
import streamlit as st
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text: str) -> list[str]:
    start = text.find("*** START")
    end = text.find("*** END")

    if start != -1 and end != -1:
        text = text[start:end]

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens

def ambil_angka(nama_file):
    angka = re.search(r'\d+', nama_file)
    if angka:
        return int(angka.group())
    return 0

@st.cache_data(show_spinner="Membaca dokumen...")
def load_documents(folder: str) -> dict:
    docs = {}

    if not os.path.isdir(folder):
        return docs
    for fname in sorted(os.listdir(folder), key=ambil_angka):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                docs[fname] = preprocess_text(f.read())
    
    return docs


@st.cache_data(show_spinner="Menghitung TF...")
def build_tf(documents: dict) -> pd.DataFrame:
    all_terms = sorted(set().union(*documents.values()))
    doc_names = list(documents.keys())

    rows = {}
    for term in all_terms:
        row = {}
        for doc in doc_names:
            tokens = documents[doc]
            total_words = len(tokens)
            count = tokens.count(term)
            row[doc] = round(count / total_words, 4) if total_words > 0 else 0 
        rows[term] = row

    return pd.DataFrame(rows).T


@st.cache_data(show_spinner="Menghitung IDF...")
def build_idf(documents: dict) -> pd.DataFrame:
    all_terms = sorted(set().union(*documents.values()))
    num_docs = len(documents)

    rows = []
    for term in all_terms:
        df_val = sum(1 for tokens in documents.values() if term in tokens)
        idf_val = round(math.log10(num_docs / df_val), 4) if df_val > 0 else 0
        rows.append({
            "Term": term,
            "DF (Jumlah Dokumen)": df_val,
            f"IDF (log({num_docs}/DF))": idf_val
        })

    return pd.DataFrame(rows).set_index("Term")

@st.cache_data(show_spinner="Menghitung TF-IDF...")
def build_tfidf(tf_df: pd.DataFrame, idf_df: pd.DataFrame) -> pd.DataFrame:
    idf_col = idf_df.columns[-1]
    tfidf_df = tf_df.copy()

    for term in tfidf_df.index:
        idf_val = idf_df.loc[term, idf_col] if term in idf_df.index else 0
        for doc in tfidf_df.columns:
            tfidf_df.loc[term, doc] = round(tfidf_df.loc[term, doc] * idf_val, 4)

    return tfidf_df

def computer_query_tables(query_terms: list[str], documents: dict, tf_df: pd.DataFrame, idf_df: pd.DataFrame, tfidf_df: pd.DataFrame):
    num_docs = len(documents)
    idf_col = idf_df.columns[-1]

    tf_rows = {}
    for term in query_terms:
        if term in tf_df.index:
            tf_rows[term] = tf_df.loc[term]
        else:
            tf_rows[term] = {doc: 0 for doc in tf_df.columns}
    tf_query = pd.DataFrame(tf_rows).T

    idf_rows = []
    for term in query_terms:
        if term in idf_df.index:
            df_val = idf_df.loc[term, "DF (Jumlah Dokumen)"]
            idf_val = idf_df.loc[term, idf_col]
        else:
            df_val = 0
            idf_val = 0
        idf_rows.append({
            "Term": term,
            "DF (Jumlah Dokumen)": df_val,
            f"IDF (log({num_docs}/DF))": idf_val
        })

    idf_query = pd.DataFrame(idf_rows).set_index("Term")

    tfidf_rows = {}
    for term in query_terms:
        if term in tfidf_df.index:
            tfidf_rows[term] = tfidf_df.loc[term]
        else:
            tfidf_rows[term] = {doc: 0 for doc in tfidf_df.columns}
    tfidf_query = pd.DataFrame(tfidf_rows).T

    return tf_query, idf_query, tfidf_query


def preprocess_query(query: str) -> list[str]:
    query = query.lower()
    query = re.sub(r'[^a-z\s]', '', query)
    tokens = query.split()

    seen = set()
    result = []

    for token in tokens:
        if token in stop_words:
            continue
        stemmed = stemmer.stem(token)
        if stemmed not in seen:
            seen.add(stemmed)
            result.append(stemmed)
    return result


st.set_page_config(page_title="TF-IDF System", page_icon=":books:", layout="wide")
st.title("📊 TF-IDF Information Retrieval System")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    folder_path = st.text_input("Folder dokumen", value="documents")
    st.markdown("---")
    st.markdown(
        "**Cara penggunaan:**\n"
        "- Masukkan beberapa kata di kolom query\n"
        "- Pisahkan antar kata dengan spasi\n"
        "- Contoh: 'love war music life'\n\n"
        "**Catatan:**\n"
        "- Stopword otomatis dibuang\n"
        "- Setiap kata di-stem menggunakan Porter Stemmer\n"
        "- Query bukan boolean - semua kata dicari"
    )

documents = load_documents(folder_path)

if not documents:
    st.warning(f"Folder '{folder_path}' tidak ditemukan atau tidak berisi file .txt.")
    st.stop()

st.success(f"✅ {len(documents)} dokumen berhasil dimuat.")

tf_df = build_tf(documents)
idf_df = build_idf(documents)
tfidf_df = build_tfidf(tf_df, idf_df)

st.subheader("🔎 Masukkan Query")
st.caption("Pisahkan setiap kata dengan spasi. Stopword akan otomatis dibuang.")
query_input = st.text_input("Contoh: 'love war music life bomb")

if query_input:
    query_terms = preprocess_query(query_input)

    if not query_terms:
        st.warning("Semua kata dalam query adalah stopword atau kosong. Coba dengan kata lain.")
    else:
        st.info(
            f"**Term setelah preprocessing** (lowercase -> hapus stopword -> stem): \n\n"
            f"`{'`, `'.join(query_terms)}`"
        )

        tf_q, idf_q, tfidf_q = computer_query_tables(query_terms, documents, tf_df, idf_df, tfidf_df)

        with st.expander("📐 Perhitungan Step-by-Step"):
            # st.markdown("---")
            st.subheader("📌 Perhitungan Term Frequency (TF)")
            st.latex(r"TF(t,\, d) = \frac{\text{jumlah kemunculan } t \text{ dalam dokumen } d}{\text{total kata dalam dokumen } d}")
            st.caption("Nilai TF menunjukkan seberapa sering sebuah term (kata) muncul dalam satu dokumen relatif terhadap panjang dokumen.")

            tf_display = tf_q.copy().astype(str)
            tf_display = tf_display.replace("0.0", "0")
            doc_names = list(documents.keys())
            for term in tf_q.index:
                for doc in doc_names:
                    tokens = documents[doc]
                    total = len(tokens)
                    count = tokens.count(term)
                    tf_val = tf_q.loc[term, doc]
                    if count > 0:
                        tf_display.loc[term, doc] = f"{count}/{total} = {tf_val}"
                    else:
                        tf_display.loc[term, doc] = "0"
            st.dataframe(tf_display, width="stretch")

            st.markdown("---")
            st.subheader("📌 Perhitungan Inverse Document Frequency (IDF)")
            num_docs = len(documents)
            st.latex(r"IDF(t) = \log_{10} \frac{num_docs}{DF(t)}")
            st.caption(
                f"**num_docs** = Jumlah total dokumen = **{num_docs}**   \n"
                "**DF(t)** = Jumlah dokumen yang mengandung term t  \n"
                "IDF tinggi -> term langka (lebih informatif). IDF rendah -> term umum di banyak dokumen"
            )

            idf_col = idf_q.columns[-1]
            idf_display = idf_q.copy().astype(str)
            idf_display = idf_display.replace("0.0", "0")
            idf_display[idf_col] = idf_display.apply(
                lambda row: (
                    f"log({num_docs}/{int(row['DF (Jumlah Dokumen)'])}) = {row[idf_col]}"
                    if row["DF (Jumlah Dokumen)"] > 0 else "0"
                ),
                axis=1
            )
            st.dataframe(idf_display, width="stretch")

        # st.markdown("---")
        st.subheader("📌 Perhitungan TF-IDF")
        st.latex(r"TF\text{-}IDF(t,\, d) = TF(t,\, d) \times IDF(t)")
        st.caption("Nilai TF-IDF tinggi -> term (kata) tersebut penting dan relatif unik untuk dokumen itu.")
        st.dataframe(tfidf_q, width="stretch")

        st.markdown("---")
        st.subheader("🏆 Ranking Dokumen")
        st.caption("Dokumen diurutkan berdasarkan jumlah skor TF-IDF dari semua term query (skor lebih tinggi = lebih relevan).")

        scores = tfidf_q.sum(axis=0).sort_values(ascending=False)
        ranking_df = pd.DataFrame({
            "Dokumen": scores.index,
            "Total Skor TF_IDF": scores.values.round(4)
        }).reset_index(drop=True)
        ranking_df.index += 1
        ranking_df.index.name = "Rank"
        st.dataframe(ranking_df, width="stretch")

        top_doc = scores.index[0]
        top_score = round(scores.iloc[0], 4)

        if top_score > 0:
            st.success(f"🥇 Dokumen paling relevan: **{top_doc}** (skor: {top_score})")
        else:
            st.info("Tidak ada dokumen yang mengandung term-term dari query ini.")

st.divider()

tab1, tab2 = st.tabs(["📊 Tabel TF (Seluruh Term)", "📊 Tabel IDF (Seluruh Term)"])

with tab1:
    st.subheader("Term Frequency - Seluruh Term")
    st.latex(r"TF(t,\, d) = \frac{\text{jumlah kemunculan } t \text{ dalam dokumen } d}{\text{total kata dalam dokumen } d}")
    st.caption(f"Menampilkan semua {len(tf_df)} term dari {len(documents)} dokumen.")

    filter_tf = st.text_input("🔍 Filter term:", key="filter_tf")
    if filter_tf:
        filtered = tf_df[tf_df.index.str.contains(filter_tf.lower(), na=False)]
        st.caption(f"{len(filtered)} term cocok dengan '{filter_tf}'.")
        st.dataframe(filtered.round(4), width="stretch", height=500)
    else:
        st.dataframe(tf_df.round(4), width="stretch", height=500)

with tab2:
    st.subheader("Inverse Document Frequency - Seluruh Term")
    num_docs = len(documents)
    st.latex(r"IDF(t) = \log_{10} \frac{num_docs}{DF(t)}")
    st.caption(f"num_docs = {num_docs} dokumen. Menampilkan semua {len(idf_df)} term.")

    filter_idf = st.text_input("🔍 Filter term:", key="filter_idf")
    if filter_idf:
        filtered = idf_df[idf_df.index.str.contains(filter_idf.lower(), na=False)]
        st.caption(f"{len(filtered)} term cocok dengan '{filter_idf}'.")
        st.dataframe(filtered, width="stretch", height=500)
    else:
        st.dataframe(idf_df, width="stretch", height=500)