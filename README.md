# Zarle Tagger

One-step question-tagging tool built with Streamlit ☁️

## Features
- Upload a JSON list of questions
- Paste a list of tags (with optional descriptions)
- Tags are embedded once; questions are embedded in batches
- Adds a `questionTags` field and lets you download the tagged file

## Run locally
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
streamlit run streamlit_tag_app.py
```