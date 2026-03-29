import streamlit as st
from PIL import Image
import pandas as pd
import re
from typing import Optional, Set, Tuple, List, Dict, Any

# Sentiment analysis methods
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Memoize transformers pipeline for efficiency
@st.cache_resource
def get_transformers_pipe():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def textblob_sentiment(text):
    tb = TextBlob(text)
    polarity = tb.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "unknown"

def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    if vs['compound'] > 0.05:
        return "positive"
    elif vs['compound'] < -0.05:
        return "negative"
    else:
        return "unknown"

def transformers_sentiment(text):
    sentiment_pipe = get_transformers_pipe()
    try:
        res = sentiment_pipe(text)[0]
        label = res['label'].lower()
        if "pos" in label:
            return "positive"
        elif "neg" in label:
            return "negative"
        else:
            return "unknown"
    except Exception:
        return "unknown"

def textblob_explanation(text):
    tb = TextBlob(text)
    polarity = tb.sentiment.polarity
    if polarity > 0.1:
        explanation = (
            f"TextBlob assigned a 'positive' label because the polarity score "
            f"was {polarity:.3f}, greater than the threshold 0.1."
        )
    elif polarity < -0.1:
        explanation = (
            f"TextBlob assigned a 'negative' label because the polarity score "
            f"was {polarity:.3f}, less than the threshold -0.1."
        )
    else:
        explanation = (
            f"TextBlob assigned 'unknown' because the polarity score was {polarity:.3f}, "
            "which is close to neutral (between -0.1 and 0.1)."
        )
    return explanation

def vader_explanation(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    if vs['compound'] > 0.05:
        explanation = (
            f"VADER classified the sentiment as 'positive' because the compound score "
            f"was {vs['compound']:.3f}, greater than the threshold 0.05."
        )
    elif vs['compound'] < -0.05:
        explanation = (
            f"VADER classified the sentiment as 'negative' because the compound score "
            f"was {vs['compound']:.3f}, less than the threshold -0.05."
        )
    else:
        explanation = (
            f"VADER classified the sentiment as 'unknown' because the compound score was {vs['compound']:.3f}, "
            "close to zero (between -0.05 and 0.05)."
        )
    return explanation

def transformers_explanation(text):
    sentiment_pipe = get_transformers_pipe()
    try:
        res = sentiment_pipe(text)[0]
        label = res['label']
        score = res['score']
        if "POSITIVE" in label:
            explanation = f"The transformers model labeled the sentiment as 'positive' with confidence {score:.2f}."
        elif "NEGATIVE" in label:
            explanation = f"The transformers model labeled the sentiment as 'negative' with confidence {score:.2f}."
        else:
            explanation = "The transformers model assigned the label 'unknown' because the result was inconclusive."
        return explanation
    except Exception:
        return "The transformers model could not analyze this text."

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", str(text).lower())

def load_lexicon(uploaded_lexicon_file) -> Tuple[Set[str], Set[str]]:
    df_lex = pd.read_csv(uploaded_lexicon_file)
    if "positive" not in df_lex.columns or "negative" not in df_lex.columns:
        raise ValueError("Lexicon CSV must have columns named 'positive' and 'negative'.")

    pos_words = (
        df_lex["positive"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )
    neg_words = (
        df_lex["negative"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )

    pos_set: Set[str] = set(w for w in pos_words.tolist() if w)
    neg_set: Set[str] = set(w for w in neg_words.tolist() if w)
    return pos_set, neg_set

def lexicon_display_name_from_filename(filename: str) -> Optional[str]:
    m = re.match(r"^(.+)_lexicon\.csv$", (filename or "").strip(), flags=re.IGNORECASE)
    if not m:
        return None
    name = m.group(1).strip()
    return name or None

def lexicon_sentiment_details(text: str, pos_set: Set[str], neg_set: Set[str], prefix: str = "Lexicon") -> Dict[str, Any]:
    tokens = _tokenize(text)
    pos_hits = [t for t in tokens if t in pos_set]
    neg_hits = [t for t in tokens if t in neg_set]

    pos_count = len(pos_hits)
    neg_count = len(neg_hits)

    if pos_count > neg_count:
        label = "positive"
    elif neg_count > pos_count:
        label = "negative"
    else:
        label = "unknown"

    return {
        f"{prefix}_label": label,
        f"{prefix}_pos_count": pos_count,
        f"{prefix}_neg_count": neg_count,
        f"{prefix}_pos_words": ", ".join(sorted(set(pos_hits))) if pos_hits else "",
        f"{prefix}_neg_words": ", ".join(sorted(set(neg_hits))) if neg_hits else "",
    }

logo_path = "cbs_logo.png"  # ensure this file exists in the same folder
col1, col2 = st.columns([1, 3])
with col1:
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=180)
    except Exception:
        st.empty()
with col2:
    st.title("The Analytics Advantage")

st.subheader("Sentiment Analysis")

use_lexicon = st.checkbox("Use a custom lexicon (CSV with 'positive' and 'negative' columns)")
uploaded_lexicon = None
pos_set: Optional[Set[str]] = None
neg_set: Optional[Set[str]] = None
lexicon_prefix = "Lexicon"
if use_lexicon:
    uploaded_lexicon = st.file_uploader("Upload lexicon CSV", type="csv")
    if uploaded_lexicon is not None:
        try:
            pos_set, neg_set = load_lexicon(uploaded_lexicon)
            inferred_name = lexicon_display_name_from_filename(getattr(uploaded_lexicon, "name", ""))
            lexicon_prefix = inferred_name or "Lexicon"
        except Exception as e:
            st.error(str(e))
            pos_set, neg_set = None, None

input_mode = st.radio("Choose input method:", ["Single text input", "CSV file upload (must contain a 'text' column)"])
analyze = st.button("Analyze sentiment")

if input_mode == "Single text input":
    user_text = st.text_area("Enter text to analyze for sentiment:")

    if analyze and user_text.strip():
        results: dict = {
            "TextBlob": textblob_sentiment(user_text),
            "VADER": vader_sentiment(user_text),
            "Transformers": transformers_sentiment(user_text)
        }
        if use_lexicon and pos_set is not None and neg_set is not None:
            results.update(lexicon_sentiment_details(user_text, pos_set, neg_set, prefix=lexicon_prefix))

        st.session_state['last_user_text'] = user_text
        st.session_state['last_results'] = results
        st.session_state['last_lexicon_loaded'] = bool(use_lexicon and pos_set is not None and neg_set is not None)
        st.session_state['last_lexicon_prefix'] = lexicon_prefix

    if 'last_results' in st.session_state:
        st.subheader("Sentiment Labels")
        st.dataframe(
            pd.DataFrame([st.session_state['last_results']]),
            hide_index=True,
            width="stretch",
        )

        explain = st.button("Explain analysis")

        if explain and 'last_user_text' in st.session_state:
            st.subheader("Method Explanations")
            explanations = {
                "TextBlob": textblob_explanation(st.session_state['last_user_text']),
                "VADER": vader_explanation(st.session_state['last_user_text']),
                "Transformers": transformers_explanation(st.session_state['last_user_text'])
            }
            lex_prefix = st.session_state.get("last_lexicon_prefix", "Lexicon")
            if st.session_state.get("last_lexicon_loaded") and f"{lex_prefix}_label" in st.session_state["last_results"]:
                lr = st.session_state["last_results"]
                explanations[lex_prefix] = (
                    f"{lex_prefix} method counted {lr.get(f'{lex_prefix}_pos_count', 0)} positive words "
                    f"and {lr.get(f'{lex_prefix}_neg_count', 0)} negative words, so sentiment was '{lr.get(f'{lex_prefix}_label')}'."
                )
            for k, v in explanations.items():
                st.markdown(f"**{k}:** {v}")

else:
    uploaded_file = st.file_uploader("Upload a CSV file (must contain a 'text' column).", type="csv")

    if analyze and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("The uploaded CSV must contain a column named 'text'.")
        else:
            df_results = df.copy()
            texts = df_results["text"].astype(str)
            df_results["TextBlob_label"] = texts.apply(textblob_sentiment)
            df_results["VADER_label"] = texts.apply(vader_sentiment)
            df_results["Transformers_label"] = texts.apply(transformers_sentiment)

            if use_lexicon and pos_set is not None and neg_set is not None:
                lex_details = texts.apply(lambda t: lexicon_sentiment_details(t, pos_set, neg_set, prefix=lexicon_prefix))
                df_lex = pd.DataFrame(list(lex_details))
                df_results = pd.concat([df_results, df_lex], axis=1)

            st.subheader("Sentiment Labels for Uploaded CSV")
            st.dataframe(df_results, hide_index=True, width="stretch")

            csv_data = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_data,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )
