# ──────────────────────────────────────────────────────────────────────────────
# streamlit_tag_app.py  – direction-aware version
# ──────────────────────────────────────────────────────────────────────────────
import json, os, re, tempfile
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import openai
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from openai import BadRequestError

# ─── Settings ────────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")          # or st.secrets["OPENAI_API_KEY"]
MODEL        = "text-embedding-3-small"
BATCH        = 64
DEFAULT_TAU  = 0.80

# ─── Page / Theme (unchanged) ────────────────────────────────────────────────
st.set_page_config("Zarle AI Automator", "🤖", "wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp { background-color:#121212; color:#EEE; }
[data-testid="stSidebar"]{background-color:#1F1F1F;padding-top:1rem;}
header{visibility:hidden;} .block-container{padding-top:0rem;}
.stFileUploader>label{width:100%;padding:1rem;background-color:#212121;
border:2px dashed #444;border-radius:8px;color:#CCC;}
button[kind="primary"]{background-color:#9C27B0!important;color:white!important;
font-weight:bold;border:none;border-radius:8px;padding:0.6em 1.4em;
transition:background-color .3s ease,transform .2s ease;}
button[kind="primary"]:hover{background-color:#BA68C8!important;transform:scale(1.03);}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar (unchanged) ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;height:200px">
        <img src="https://raw.githubusercontent.com/rishuSingh404/Zarle/main/logo.png" width="150">
    </div>
    <div style="color:white">
        <h3 style="margin-bottom:.2em">Zarle AI Automator</h3>
        <p style="margin-top:0">Fast tagging of question JSON files with AI embeddings.</p>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Tag Questions"],
        icons=["tags"],
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0", "background-color": "#1F1F1F"},
            "icon": {"font-size": "20px", "color": "#9C27B0"},
            "nav-link": {"font-size": "16px", "color": "#ECECEC", "text-align": "left"},
            "nav-link-selected": {"background-color": "#9C27B0", "color": "#FFF", "font-weight": "bold"},
        },
    )

# ─── Tag helpers (unchanged) ────────────────────────────────────────────────
TAG_SPLIT_RE  = re.compile(r"[,\|;/\n]+")
PAIR_RE       = re.compile(r"\s*[:=]\s*")

def parse_tags(text: str) -> List[Dict[str, str]]:
    tokens = [t.strip() for t in TAG_SPLIT_RE.split(text) if t.strip()]
    tags   = []
    for tok in tokens:
        if PAIR_RE.search(tok):
            title, desc = PAIR_RE.split(tok, 1)
        else:
            title = desc = tok
        tags.append({"title": title.strip(), "description": desc.strip()})
    return tags

# ─── NEW: flatten standalone + direction-based questions ────────────────────
def flatten_questions(data: List[Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts:
      { 'obj': <reference to question dict>,
        'text': <paragraph + question text for embedding> }
    """
    flat = []
    for item in data:
        # Direction/passage set
        if isinstance(item, dict) and isinstance(item.get("questions"), list):
            para = item.get("paragraph", "")
            for sub in item["questions"]:
                q_text = sub.get("question", "")
                combined = f"{para}\n\n{q_text}" if para else q_text
                flat.append({"obj": sub, "text": combined})
        else:  # standalone
            if isinstance(item, dict):
                flat.append({"obj": item, "text": item.get("question", "")})
    return flat

# ─── Embedding helpers (unchanged safeguards) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def embed_texts(texts: List[str]) -> np.ndarray:
    cleaned = []
    for t in texts:
        if t is None:
            cleaned.append(" ")
        else:
            s = str(t).strip()
            cleaned.append(s if s else " ")
    try:
        resp = openai.embeddings.create(input=cleaned, model=MODEL)
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    except BadRequestError as e:
        if "maximum context length" in str(e) and len(cleaned) > 1:
            mid = len(cleaned) // 2
            return np.vstack([embed_texts(cleaned[:mid]), embed_texts(cleaned[mid:])])
        raise

def choose_tags(q_vec, tag_vecs, tags, threshold):
    sims = cosine_similarity([q_vec], tag_vecs)[0]
    idxs = [i for i, s in enumerate(sims) if s >= threshold] or [int(np.argmax(sims))]
    idxs.sort(key=lambda i: sims[i], reverse=True)
    return [tags[i]["title"] for i in idxs]

# ─── Main interface ─────────────────────────────────────────────────────────
if selected == "Tag Questions":
    st.header("🏷️  Tag Questions JSON")

    q_file  = st.file_uploader("Upload questions JSON", type="json")
    tag_text = st.text_area(
        "Enter tag list (comma / ; / | / newline separated).\n"
        "Optional description with `:` or `=` (e.g.  Math = mathematics questions)",
        height=160,
    )

    threshold = st.slider("Cosine-similarity threshold (τ)", 0.00, 1.00, DEFAULT_TAU, 0.01)
    run = st.button("Generate tagged JSON  ⏩", type="primary")

    if run:
        # Preconditions
        if not openai.api_key:
            st.error("❌  OPENAI_API_KEY not set.")
            st.stop()
        if not (q_file and tag_text.strip()):
            st.warning("Please supply both a JSON file and at least one tag.")
            st.stop()

        tags = parse_tags(tag_text)
        st.success(f"Loaded {len(tags)} tags.")

        # Load JSON
        try:
            data = json.load(q_file)
        except Exception as e:
            st.error(f"Error reading JSON: {e}")
            st.stop()
        if not isinstance(data, list):
            st.error("JSON root must be a list.")
            st.stop()

        # Flatten questions
        flat = flatten_questions(data)
        if not flat:
            st.error("No question objects found in the JSON.")
            st.stop()

        # Embed tags once
        with st.spinner("Embedding tags…"):
            tag_vecs = embed_texts([t["description"] for t in tags])

        # Embed questions & tag
        progress = st.progress(0)
        for start in range(0, len(flat), BATCH):
            texts_batch = [d["text"] for d in flat[start:start+BATCH]]
            vecs = embed_texts(texts_batch)
            for d, v in zip(flat[start:start+BATCH], vecs):
                d["obj"]["questionTags"] = choose_tags(v, tag_vecs, tags, threshold)
            progress.progress(min((start+BATCH)/len(flat), 1.0))

        # Write output
        tmp_out = tempfile.NamedTemporaryFile(
            delete=False, suffix="_tagged.json", mode="w", encoding="utf-8"
        )
        json.dump(data, tmp_out, ensure_ascii=False, indent=2)
        tmp_out.close()

        st.success("✅  Tagging complete!")
        st.markdown("**Preview (first 3 questions found):**")
        st.json([d["obj"] for d in flat[:3]])

        with open(tmp_out.name, "rb") as f:
            st.download_button(
                "⬇️  Download tagged JSON",
                data=f,
                file_name=Path(q_file.name).stem + "_tagged.json",
                mime="application/json",
            )
