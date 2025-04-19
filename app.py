import streamlit as st
import spacy
try:
    spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    spacy.load("en_core_web_sm")

from summarizer import TextSummarizer

st.set_page_config(page_title="NLP Text Summarizer", page_icon="📝", layout="wide")

st.title("📝 Robust NLP Text Summarizer")

st.markdown(
    """
Upload a **PDF**, paste some **text**, or provide a **URL**.  
Tune the summary length from the sidebar and get both **extractive** and **abstractive** summaries.
"""
)

# Sidebar controls
st.sidebar.header("⚙️ Controls")
num_sentences = st.sidebar.slider("Sentences (extractive)", 1, 15, 5)
min_len = st.sidebar.slider("Minimum tokens (abstractive)", 20, 300, 80)
max_len = st.sidebar.slider("Maximum tokens (abstractive)", 50, 512, 200)

# Input selector
option = st.selectbox("Choose input type", ["Text", "PDF", "URL"])

summarizer = TextSummarizer()

if option == "Text":
    user_text = st.text_area("Enter text to summarize:", height=300)
    if st.button("Summarize Text", type="primary"):
        if len(user_text.split()) < 10:
            st.warning("⚠️ Please provide at least 10 words for meaningful summarization.")
        else:
            with st.spinner("Summarizing …"):
                extractive = summarizer.extractive_summary(user_text, num_sentences)
                abstractive = summarizer.summarize_long_text(user_text, min_length=min_len, max_length=max_len)
            st.success("Done!")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Extractive Summary")
                st.write(extractive)
            with col2:
                st.subheader("Abstractive Summary")
                st.write(abstractive)

elif option == "PDF":
    file = st.file_uploader("Upload PDF", type=["pdf"])
    if file and st.button("Summarize PDF", type="primary"):
        with st.spinner("Reading & Summarizing PDF …"):
            pdf_summary = summarizer.summarize_pdf(file, min_length=min_len, max_length=max_len)
        st.subheader("PDF Summary")
        st.write(pdf_summary)

elif option == "URL":
    url = st.text_input("Enter URL:")
    if st.button("Summarize URL", type="primary") and url:
        with st.spinner("Fetching & Summarizing URL …"):
            url_summary = summarizer.summarize_url(url, min_length=min_len, max_length=max_len)
        st.subheader("URL Summary")
        st.write(url_summary)

st.markdown("---")
st.caption("Built with spaCy, Hugging Face Transformers, and Streamlit.")
