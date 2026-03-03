# app/feature_engineering.py
import re
import pandas as pd
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around)

# --- Text cleaning (matches your notebook intent) ---
def clean_text_keep_emojis(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = re.sub(r"http\S+|www\.\S+", " ", t)      # remove urls
    t = re.sub(r"[\r\n\t]+", " ", t)            # normalize whitespace
    t = re.sub(r"\s{2,}", " ", t)               # collapse spaces
    return t.strip()

# --- Sentiment markers (from your notebook) ---
positive_en = set("good great best amazing awesome excellent love loved nice perfect works working helpful fast reliable".split())
negative_en = set("bad worst terrible hate hated slow bug bugs buggy crash crashes crashing error errors fail failed failing useless poor disappointed disappointing scam".split())

positive_sw = set(["nzuri", "safi", "bomba", "vizuri", "bora", "napenda"])
negative_sw = set(["mbaya", "mbovu", "kosa", "makosa", "tatizo", "matatizo", "inakwama", "haifanyi", "haiwezi"])

positive_sh = set(["poa", "noma", "fresh", "safi", "kali", "fiti"])
negative_sh = set(["mbaya", "mbovu", "imekwama", "imeshindwa", "imeniboo"])

pos_emojis = {"😀","😃","😄","😁","😍","😊","👍","❤️","❤","💯","🔥","🎉"}
neg_emojis = {"😡","😠","😤","😭","😢","👎","💔","😔","🤬"}

negation_words = {"not", "no", "never", "hakuna", "sio", "si"}

def sentiment_score_improved(text: str) -> float:
    if not isinstance(text, str):
        return 0.0

    t = text.lower()
    words = re.findall(r"[a-zA-ZÀ-ÿ']+", t)
    if len(words) == 0:
        return 0.0

    score = 0
    negate = False

    for w in words:
        if w in negation_words:
            negate = True
            continue

        val = 0
        if w in positive_en or w in positive_sw or w in positive_sh:
            val = 1
        elif w in negative_en or w in negative_sw or w in negative_sh:
            val = -1

        if negate:
            val = -val
            negate = False

        score += val

    # emoji sentiment
    for e in pos_emojis:
        if e in text:
            score += 1
    for e in neg_emojis:
        if e in text:
            score -= 1

    return score / len(words)

def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df["review_text"] = df["review_text"].fillna("").astype(str)

    # rating is validated by pydantic, but keep safe conversion
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    df["rating"] = df["rating"].fillna(3).astype(int)  # neutral fallback instead of 0

    df["thumbs_up"] = pd.to_numeric(df.get("thumbs_up", 0), errors="coerce").fillna(0).astype(int)

    # safer boolean handling
    df["is_code_mixed"] = df.get("is_code_mixed", False).fillna(False).astype(bool).astype(int)
    df["is_sheng_like"] = df.get("is_sheng_like", False).fillna(False).astype(bool).astype(int)

    df["text_clean"] = df["review_text"].map(clean_text_keep_emojis)
    df["text_norm"] = (df["text_clean"].str.lower().str.replace(r"\s+", " ", regex=True).fillna(""))

    df["word_count"] = df["text_norm"].astype(str).str.split().apply(len)
    df["exclamation_count"] = df["text_norm"].astype(str).str.count("!")

    # emoji count aligned with your emoji sets
    emoji_set = pos_emojis.union(neg_emojis)
    df["emoji_count"] = df["review_text"].apply(lambda t: sum(1 for ch in str(t) if ch in emoji_set))

    df["sent_score_norm"] = df["text_norm"].map(sentiment_score_improved)
    df["rating_scaled"] = (df["rating"] - 3) / 2
    df["rating_sentiment_mismatch"] = (df["rating_scaled"] - df["sent_score_norm"]) ** 2
    df["abs_sentiment_strength"] = df["sent_score_norm"].abs()

    feature_cols = [
        "text_norm", "rating", "thumbs_up", "is_code_mixed", "is_sheng_like",
        "word_count", "emoji_count", "exclamation_count", "sent_score_norm",
        "rating_scaled", "rating_sentiment_mismatch", "abs_sentiment_strength",
    ]
    return df[feature_cols]
