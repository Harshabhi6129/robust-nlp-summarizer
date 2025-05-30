"""
Updated text_summarizer.py
Uses beautifulsoup4 + requests for URL summarization
Fully Streamlit Cloud compatible ✅
"""

from __future__ import annotations

import functools
import heapq
from typing import List

import spacy
import torch
import requests
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
import pdfplumber

__all__ = ["TextSummarizer"]

#############################
# Heavy resource loaders    #
#############################

@functools.lru_cache(maxsize=1)
def _load_spacy(model_name: str = "en_core_web_sm"):
    return spacy.load(model_name)

@functools.lru_cache(maxsize=1)
def _load_bart(model_name: str = "facebook/bart-large-cnn"):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model

#############################
# Core Summarizer Class     #
#############################

class TextSummarizer:
    """Extractive + Abstractive summarization class."""

    def __init__(self, spacy_model: str = "en_core_web_sm", bart_model: str = "facebook/bart-large-cnn") -> None:
        self.nlp = _load_spacy(spacy_model)
        self.tokenizer, self.model = _load_bart(bart_model)

    def extractive_summary(self, text: str, num_sentences: int = 5) -> str:
        doc = self.nlp(text)
        sentences = [s.text for s in doc.sents if s.text.strip()]
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        word_freq: dict[str, float] = {}
        for token in doc:
            if not (token.is_stop or token.is_punct):
                lemma = token.lemma_.lower()
                word_freq[lemma] = word_freq.get(lemma, 0) + 1
        if not word_freq:
            return " ".join(sentences[:num_sentences])
        max_freq = max(word_freq.values())
        word_freq = {w: f / max_freq for w, f in word_freq.items()}

        scores: dict[str, float] = {}
        for sent in sentences:
            token_scores = sum(word_freq.get(tok.lemma_.lower(), 0.0) for tok in self.nlp(sent))
            scores[sent] = token_scores
        top_sentences = heapq.nlargest(num_sentences, scores, key=scores.get)
        return " ".join(top_sentences)

    def abstractive_summary(self, text: str, *, min_length: int = 80, max_length: int = 200) -> str:
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
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def _split_chunks(self, text: str, max_tokens: int = 1024) -> List[str]:
        sentences = list(self.nlp(text).sents)
        chunks, cur, cur_len = [], [], 0
        for sent in sentences:
            token_len = len(self.tokenizer.encode(sent.text))
            if cur_len + token_len > max_tokens:
                chunks.append(" ".join(cur))
                cur, cur_len = [sent.text], token_len
            else:
                cur.append(sent.text)
                cur_len += token_len
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    def summarize_long_text(self, text: str, *, min_length: int = 80, max_length: int = 200) -> str:
        chunks = self._split_chunks(text)
        partials = [self.abstractive_summary(chunk, min_length=min_length, max_length=max_length) for chunk in chunks]
        if len(partials) > 1:
            joined = " ".join(partials)
            return self.abstractive_summary(joined, min_length=min_length // 2, max_length=max_length)
        return partials[0]

    def summarize_pdf(self, file_buffer, *, min_length: int = 80, max_length: int = 200):
        try:
            with pdfplumber.open(file_buffer) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            return f"❌ Could not read PDF: {e}"
        if not text.strip():
            return "No extractable text found in the PDF."
        return self.summarize_long_text(text, min_length=min_length, max_length=max_length)

    def summarize_url(self, url: str, *, min_length: int = 80, max_length: int = 200):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return f"❌ Failed to fetch URL (status code {response.status_code})"
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all('p')
            text = " ".join([para.get_text() for para in paragraphs])
        except Exception as e:
            return f"❌ Error fetching URL: {e}"
        if not text.strip():
            return "No main article content found on this URL."
        return self.summarize_long_text(text, min_length=min_length, max_length=max_length)
