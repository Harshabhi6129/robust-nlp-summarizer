# Robust NLP Text Summarizer

A Streamlit application that combines frequency‑based *extractive* summarization and BART‑based *abstractive* summarization. Supports raw text, PDFs, and web articles.

---

## Quick Start

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Project Structure

```
.
├── app.py                   # Streamlit front‑end
├── summarizer/
│   ├── __init__.py
│   └── text_summarizer.py   # Core summarization logic
├── requirements.txt
└── README.md
```

## Features

* **Extractive summary** using spaCy word frequencies
* **Abstractive summary** with Facebook BART (large‑cnn)
* PDF and URL ingestion with graceful error handling
* GPU support out‑of‑the‑box (CUDA detected automatically)
* Sidebar controls for sentence count and abstractive length

Enjoy! 🎉

