# ──────────────────────────────────────────────────────────────────────────────
# streamlit_tag_app.py
# One-step JSON question-tagger with the same look-and-feel as Zarle AI Automator
# ──────────────────────────────────────────────────────────────────────────────
import json, os, re, tempfile
from pathlib import Path
from typing import List, Dict

import numpy as np
import openai
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu

# ─── Settings ────────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")               # or st.secrets["OPENAI_API_KEY"]
MODEL        = "text-embedding-3-small"
BATCH        = 256                                         # question batch size
DEFAULT_DELTA = 0.05                                       # similarity window

# ─── Page / Theme ────────────────────────────────────────────────────────────
st.set_page_config("Zarle AI Automator", "🤖", "wide", initial_sidebar_state="expanded")

# Global CSS (identical to your 5-step app) ───────────────────────────────────
st.markdown("""
<style>
/* dark background + typography */
.stApp { background-color:#121212; color:#EEE; }
/* sidebar */
[data-testid="stSidebar"]{background-color:#1F1F1F;padding-top:1rem;}
/* hide default header padding */
header{visibility:hidden;} .block-container{padding-top:0rem;}
/* file-uploader */
.stFileUploader>label{width:100%;padding:1rem;background-color:#212121;
border:2px dashed #444;border-radius:8px;color:#CCC;}
/* primary buttons */
button[kind="primary"]{background-color:#9C27B0!important;color:white!important;
font-weight:bold;border:none;border-radius:8px;padding:0.6em 1.4em;
transition:background-color .3s ease,transform .2s ease;}
button[kind="primary"]:hover{background-color:#BA68C8!important;transform:scale(1.03);}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar (logo + single menu item) ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;height:200px">
        <img src="https://raw.githubusercontent.com/rishuSingh404/Zarle/main/logo.png" width="150">
    </div>
    <div style="color:white">
        <h3 style="margin-bottom:.2em">Zarle AI Automator</h3>
        <p style="margin-top:0">Fast tagging of question JSON files with AI-generated embeddings.</p>
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

# ─── Helpers ────────────────────────────────────────────────────────────────
TAG_SPLIT_RE  = re.compile(r"[,\|;/\n]+")
PAIR_RE       = re.compile(r"\s*[:=]\s*")

def _save_temp(uploaded, suffix: str) -> Path:
    """Write uploaded file to a NamedTemporaryFile and return its Path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()
    return Path(tmp.name)

def parse_tags(text: str) -> List[Dict[str, str]]:
    """Parse multiline / delimited tag input into [{'title':..,'description':..}, ...]"""
    tokens = [t.strip() for t in TAG_SPLIT_RE.split(text) if t.strip()]
    tags   = []
    for tok in tokens:
        if PAIR_RE.search(tok):
            title, desc = PAIR_RE.split(tok, 1)
        else:
            title = desc = tok
        tags.append({"title": title.strip(), "description": desc.strip()})
    return tags

@st.cache_resource(show_spinner=False)  # reuse embeddings across runs
def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts via OpenAI and return a 2-D numpy array."""
    resp = openai.embeddings.create(input=texts, model=MODEL)
    vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def choose_tags(q_vec: np.ndarray, tag_vecs: np.ndarray, tags: List[Dict[str, str]], delta: float) -> List[str]:
    sims  = cosine_similarity([q_vec], tag_vecs)[0]
    best  = sims.max()
    idxs  = [i for i, s in enumerate(sims) if best - s <= delta] or [int(np.argmax(sims))]
    idxs.sort(key=lambda i: sims[i], reverse=True)
    return [tags[i]["title"] for i in idxs]

# ─── Main Interface ─────────────────────────────────────────────────────────
if selected == "Tag Questions":
    st.header("🏷️  Tag Questions JSON")

    q_file = st.file_uploader("Upload questions JSON", type="json")
    tag_text = st.text_area(
        "Enter tag list (comma / ; / | / newline separated).\n"
        "Optional description with `:` or `=` (e.g.  *Math = questions on mathematics*)",
        height=160,
    )
    delta = st.slider("Similarity window (Δ)", 0.0, 0.5, DEFAULT_DELTA, 0.01)
    run = st.button("Generate tagged JSON  ⏩", type="primary")

    if run:
        if not openai.api_key:
            st.error("❌  OpenAI API key not found. Please set `OPENAI_API_KEY`.")
            st.stop()
        if not (q_file and tag_text.strip()):
            st.warning("Please provide **both** the JSON file *and* at least one tag.")
            st.stop()

        # 1️⃣  Parse inputs
        tags = parse_tags(tag_text)
        st.success(f"Loaded {len(tags)} tags.")

        try:
            questions = json.load(q_file)
        except Exception as e:
            st.error(f"❌  Could not read JSON file: {e}")
            st.stop()

        if not isinstance(questions, list):
            st.error("❌  JSON root must be a list of question objects.")
            st.stop()

        # 2️⃣  Embed tags once
        with st.spinner("Embedding tags…"):
            tag_vecs = embed_texts([t["description"] for t in tags])

        # 3️⃣  Embed questions in batches & tag
        q_texts = [q.get("question", "") for q in questions]
        all_tag_lists = []
        progress = st.progress(0)
        for i in range(0, len(q_texts), BATCH):
            vecs = embed_texts(q_texts[i : i + BATCH])
            for v in vecs:
                all_tag_lists.append(choose_tags(v, tag_vecs, tags, delta))
            progress.progress(min((i + BATCH) / len(q_texts), 1.0))

        # 4️⃣  Attach tags & stream out file
        for q, tl in zip(questions, all_tag_lists):
            q["questionTags"] = tl

        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix="_tagged.json")
        json.dump(questions, tmp_out, ensure_ascii=False, indent=2)
        tmp_out.close()

        st.success("✅  Tagging complete!")
        st.markdown("**Preview (first 3 questions):**")
        st.json(questions[:3])

        with open(tmp_out.name, "rb") as f:
            st.download_button(
                "⬇️  Download tagged JSON",
                data=f,
                file_name=Path(q_file.name).stem + "_tagged.json",
                mime="application/json",
            )
