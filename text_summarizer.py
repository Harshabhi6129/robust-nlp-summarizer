import streamlit as st
import spacy
import heapq
import torch
import pdfplumber
from newspaper import Article
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List

#########################
# Cached heavy resources #
#########################

@st.cache_resource(show_spinner=False)
def load_spacy(model_name: str = "en_core_web_sm"):
    """Load spaCy model once and reuse."""
    return spacy.load(model_name)

@st.cache_resource(show_spinner=False)
def load_bart(model_name: str = "facebook/bart-large-cnn"):
    """Load BART tokenizer & model once and reuse (GPU‑aware)."""
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Use GPU if available for faster inference
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

#########################
# Summarizer class       #
#########################

class TextSummarizer:
    """Combined extractive + abstractive summarizer."""

    def __init__(self, spacy_model="en_core_web_sm", bart_model="facebook/bart-large-cnn"):
        self.nlp = load_spacy(spacy_model)
        self.tokenizer, self.model = load_bart(bart_model)

    # ---------- Extractive summarization ---------- #
    def _sentence_scores(self, doc):
        """Compute sentence scores using normalized word frequencies."""
        word_freq = {}
        for token in doc:
            if not (token.is_stop or token.is_punct):
                lemma = token.lemma_.lower()
                word_freq[lemma] = word_freq.get(lemma, 0) + 1
        if not word_freq:
            return {}
        max_freq = max(word_freq.values())
        for w in word_freq:
            word_freq[w] /= max_freq

        sentence_scores = {}
        for sent in doc.sents:
            score = 0.0
            for token in sent:
                score += word_freq.get(token.lemma_.lower(), 0.0)
            sentence_scores[sent.text] = score
        return sentence_scores

    def extractive_summary(self, text: str, num_sentences: int = 5) -> str:
        doc = self.nlp(text)
        sentences = [s.text for s in doc.sents if s.text.strip()]
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        scores = self._sentence_scores(doc)
        best_sentences = heapq.nlargest(num_sentences, scores, key=scores.get)
        return " ".join(best_sentences)

    # ---------- Abstractive summarization ---------- #
    def abstractive_summary(self, text: str, min_length: int = 80, max_length: int = 200) -> str:
        if not text:
            return ""
        inputs = self.tokenizer.encode(
            "summarize: " + text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    # ---------- Helper for long texts ---------- #
    def _split_into_chunks(self, text: str, max_tokens: int = 1024) -> List[str]:
        """Split text by sentences keeping each chunk under max_tokens."""
        sentences = list(self.nlp(text).sents)
        chunks, current_chunk = [], []
        current_len = 0
        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent.text))
            if current_len + sent_tokens > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent.text]
                current_len = sent_tokens
            else:
                current_chunk.append(sent.text)
                current_len += sent_tokens
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def summarize_long_text(self, text: str, min_length: int, max_length: int) -> str:
        chunks = self._split_into_chunks(text)
        partial_summaries = [self.abstractive_summary(chunk, min_length, max_length) for chunk in chunks]
        # Optionally run a second pass to condense partial summaries
        if len(partial_summaries) > 1:
            joined = " ".join(partial_summaries)
            return self.abstractive_summary(joined, min_length // 2, max_length)
        return partial_summaries[0]

    # ---------- PDF and URL ---------- #
    def summarize_pdf(self, file_buffer, min_length, max_length):
        try:
            with pdfplumber.open(file_buffer) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            return f"❌ Could not read PDF: {e}"
        if not text.strip():
            return "No extractable text found in the PDF."
        return self.summarize_long_text(text, min_length, max_length)

    def summarize_url(self, url: str, min_length, max_length):
        article = Article(url)
        try:
            article.download()
            article.parse()
        except Exception as e:
            return f"❌ Failed to download/parse the URL: {e}"
        if not article.text.strip():
            return "No main article content found on this URL."
        return self.summarize_long_text(article.text, min_length, max_length)

#########################
# Streamlit UI           #
#########################

st.set_page_config(page_title="NLP Text Summarizer", page_icon="📝", layout="wide")

st.title("📝 Robust NLP Text Summarizer")
st.markdown("""Use **extractive** (sentence‑ranking) and **abstractive** (BART) summarization in one place. Upload a *PDF*, paste *text*, or give a *URL* and tune the summary length with the sliders on the sidebar.""")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
num_sentences = st.sidebar.slider("Sentences (extractive)", 1, 15, 5)
min_len = st.sidebar.slider("Minimum tokens (abstractive)", 20, 300, 80)
max_len = st.sidebar.slider("Maximum tokens (abstractive)", 50, 512, 200)

option = st.selectbox("Choose an input type", ["Text", "PDF", "URL"])

summarizer = TextSummarizer()

# ------------------- Text option ------------------- #
if option == "Text":
    user_input = st.text_area("Enter text to summarize:", height=250)
    if st.button("Summarize Text", type="primary"):
        if len(user_input.split()) < 10:
            st.warning("⚠️ Text is too short for meaningful summarization.")
        else:
            with st.spinner("Generating summaries …"):
                extractive = summarizer.extractive_summary(user_input, num_sentences=num_sentences)
                abstractive = summarizer.summarize_long_text(user_input, min_len, max_len)
            st.success("Done!")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Extractive Summary")
                st.write(extractive)
                st.caption(f"Original length: {len(user_input.split())} words | Summary length: {len(extractive.split())} words")
            with col2:
                st.subheader("Abstractive Summary")
                st.write(abstractive)
                st.caption(f"Summary length: {len(abstractive.split())} words")

# ------------------- PDF option ------------------- #
elif option == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None and st.button("Summarize PDF", type="primary"):
        with st.spinner("Reading & summarizing PDF …"):
            pdf_summary = summarizer.summarize_pdf(uploaded_file, min_len, max_len)
        st.subheader("PDF Summary")
        st.write(pdf_summary)

# ------------------- URL option ------------------- #
elif option == "URL":
    url_input = st.text_input("Enter a URL to summarize:")
    if st.button("Summarize URL", type="primary") and url_input:
        with st.spinner("Fetching & summarizing URL …"):
            url_summary = summarizer.summarize_url(url_input, min_len, max_len)
        st.subheader("URL Summary")
        st.write(url_summary)

#########################
# Footer                #
#########################

st.markdown("---")
st.markdown("Made with ❤️ using spaCy, Hugging Face Transformers, and Streamlit.")
