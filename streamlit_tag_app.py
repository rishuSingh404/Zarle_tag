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
openai.api_key = os.getenv("OPENAI_API_KEY")  # or st.secrets["OPENAI_API_KEY"]
MODEL        = "text-embedding-3-small"
BATCH        = 64
DEFAULT_TAU  = 0.35

# ─── Page / Theme (unchanged) ────────────────────────────────────────────────
st.set_page_config("Zarle AI Automator", "🤖", "wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp { background-color:#121212; color:#EEE; }
[data-testid="stSidebar"]{background-color:#1F1F1F;padding-top:1rem;}
header{visibility:hidden;} .block-container{padding-top:0rem;}
.stFileUploader>label{width:100%;padding:1rem;background-color:#212121;
border:2px dashed #444;border-radius:8px;color:#CCC;}
button[kind="primary"]{background-color:#9C27B0!important;color:white!important;
font-weight:bold;border:none;border-radius:8px;padding:0.6em 1.4em;
transition:background-color .3s ease,transform .2s ease;}
button[kind="primary"]:hover{background-color:#BA68C8!important;transform:scale(1.03);}
</style>""", unsafe_allow_html=True)

# ─── Sidebar (unchanged) ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="display:flex;flex-direction:column;align-items:center;height:200px">
        <img src="https://raw.githubusercontent.com/rishuSingh404/Zarle/main/logo.png" width="150">
    </div>
    <div style="color:white">
        <h3 style="margin-bottom:.2em">Zarle AI Automator</h3>
        <p style="margin-top:0">Fast tagging of question JSON files with AI embeddings.</p>
    </div>""", unsafe_allow_html=True)

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

# ─── Tag‐group parsing & helpers ─────────────────────────────────────────────
PAIR_RE = re.compile(r"\s*[:=]\s*")

def parse_tag_groups(raw: str) -> List[List[Dict[str, str]]]:
    """
    Split on lines containing only '+', producing a list of tag‐groups.
    Each non-'+' line is parsed into {title, description}.
    """
    lines = [ln.strip() for ln in raw.splitlines()]
    groups: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []

    for ln in lines:
        if ln == '+':
            # cut a group here
            groups.append(current)
            current = []
        elif ln:
            parts = PAIR_RE.split(ln, 1)
            if len(parts) == 2:
                title, desc = parts
            else:
                title = desc = parts[0]
            current.append({"title": title.strip(), "description": desc.strip()})
        # else: empty line → ignore

    # append final group
    groups.append(current)
    return groups

@st.cache_resource(show_spinner=False)
def embed_texts(texts: List[str]) -> np.ndarray:
    cleaned = [(t or "").strip() or " " for t in texts]
    try:
        resp = openai.embeddings.create(input=cleaned, model=MODEL)
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    except BadRequestError as e:
        if "maximum context length" in str(e) and len(cleaned) > 1:
            mid = len(cleaned)//2
            return np.vstack([embed_texts(cleaned[:mid]), embed_texts(cleaned[mid:])])
        raise

def choose_tags_by_groups(
    q_vec: np.ndarray,
    tag_group_vecs: List[np.ndarray],
    tag_groups: List[List[Dict[str, str]]],
    threshold: float
) -> List[str]:
    """
    For each tag-group:
      1. Compute sims vs each tag in the group.
      2. Pick all above-threshold (or if none, the single top).
      3. Sort by descending similarity.
    Then trim each group’s picks to the size of the smallest group.
    """
    group_picks: List[List[str]] = []

    # 1. within-group selection & fallback
    for vecs, group in zip(tag_group_vecs, tag_groups):
        sims = cosine_similarity([q_vec], vecs)[0]
        idxs = [i for i,s in enumerate(sims) if s >= threshold]
        if not idxs:
            idxs = [int(np.argmax(sims))]
        # sort descending
        idxs.sort(key=lambda i: sims[i], reverse=True)
        group_picks.append([group[i]["title"] for i in idxs])

    # 2. equalize count across groups
    k = min(len(picks) for picks in group_picks)
    selected = []
    for picks in group_picks:
        selected.extend(picks[:k])
    return selected

# ─── Question flattening (unchanged) ────────────────────────────────────────
def flatten_questions(data: List[Any]) -> List[Dict[str, Any]]:
    flat = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("questions"), list):
            para = item.get("paragraph", "")
            for sub in item["questions"]:
                q_text = sub.get("question", "")
                combined = f"{para}\n\n{q_text}" if para else q_text
                flat.append({"obj": sub, "text": combined})
        else:
            if isinstance(item, dict):
                flat.append({"obj": item, "text": item.get("question", "")})
    return flat

# ─── Main interface ─────────────────────────────────────────────────────────
if selected == "Tag Questions":
    st.header("🏷️  Tag Questions JSON")

    q_file   = st.file_uploader("Upload questions JSON", type="json")
    tag_text = st.text_area(
        "Enter tag groups, separated by lines of `+`. " +
        "Within each group, one tag per line; optional `title: description`.",
        height=200,
    )
    threshold = st.slider("Cosine‐similarity threshold (τ)", 0.0, 1.0, DEFAULT_TAU, 0.01)
    run       = st.button("Generate tagged JSON  ⏩", type="primary")

    if run:
        # Preconditions
        if not openai.api_key:
            st.error("❌  OPENAI_API_KEY not set."); st.stop()
        if not (q_file and tag_text.strip()):
            st.warning("Please supply both a JSON file and at least one tag group."); st.stop()

        # 1. Parse & validate groups
        tag_groups = parse_tag_groups(tag_text)
        if any(len(g)==0 for g in tag_groups):
            st.error("Each tag group must have at least one tag. " +
                     "Check your '+' separators and non-empty lines."); st.stop()
        st.success(f"Loaded {len(tag_groups)} tag-groups.")

        # 2. Load questions JSON
        try:
            data = json.load(q_file)
        except Exception as e:
            st.error(f"Error reading JSON: {e}"); st.stop()
        if not isinstance(data, list):
            st.error("JSON root must be a list."); st.stop()

        # 3. Flatten questions
        flat = flatten_questions(data)
        if not flat:
            st.error("No question objects found in the JSON."); st.stop()

        # 4. Embed each group’s descriptions once
        with st.spinner("Embedding tags…"):
            tag_group_vecs = [
                embed_texts([t["description"] for t in grp])
                for grp in tag_groups
            ]

        # 5. Embed questions batch-wise & assign
        progress = st.progress(0)
        for i in range(0, len(flat), BATCH):
            batch = flat[i:i+BATCH]
            texts = [d["text"] for d in batch]
            q_vecs = embed_texts(texts)

            for d, qv in zip(batch, q_vecs):
                d["obj"]["questionTags"] = choose_tags_by_groups(
                    qv, tag_group_vecs, tag_groups, threshold
                )

            progress.progress(min((i+BATCH)/len(flat), 1.0))

        # 6. Output
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_tagged.json",
                                          mode="w", encoding="utf-8")
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.close()

        st.success("✅  Tagging complete!")
        st.markdown("**Preview (first 3 questions):**")
        st.json([d["obj"] for d in flat[:3]])

        with open(tmp.name, "rb") as f:
            st.download_button(
                "⬇️  Download tagged JSON",
                data=f,
                file_name=Path(q_file.name).stem + "_tagged.json",
                mime="application/json",
            )
