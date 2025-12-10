# src/preprocess.py

from pathlib import Path
import pandas as pd
import re
import nltk
import string

# Download stopwords
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

# ------------------------------
# Path Helpers
# ------------------------------
def project_root():
    return Path(__file__).resolve().parents[1]

def default_data_path():
    return project_root() / "data" / "train.csv"

# ------------------------------
# Detect CSV or TSV separator
# ------------------------------
def detect_sep(p):
    try:
        text = p.read_text(errors="ignore")[:2000]
        if "\t" in text and text.count("\t") > text.count(","):
            return "\t"
        return ","
    except:
        return ","

# ------------------------------
# Load raw file
# ------------------------------
def load_raw_df(path=None):
    p = Path(path) if path else default_data_path()
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    sep = detect_sep(p)
    df = pd.read_csv(p, sep=sep, header=None, dtype=str, engine="python", on_bad_lines='skip')
    return df

# ------------------------------
# Fix column names
# ------------------------------
def normalize_columns(df):
    if df.shape[1] >= 3:
        df = df.iloc[:, :3]
        df.columns = ["rating", "summary", "review"]
    elif df.shape[1] == 2:
        df.columns = ["rating", "review"]
        df["summary"] = ""
        df = df[["rating", "summary", "review"]]
    else:
        df["review"] = df.iloc[:, 0]
        df["rating"] = None
        df["summary"] = ""
        df = df[["rating", "summary", "review"]]

    return df

# ------------------------------
# Convert rating to sentiment label
# ------------------------------
def convert_rating_to_label(df):
    df = df.copy()

    try:
        nums = pd.to_numeric(df["rating"], errors="coerce")
        unique = set(nums.dropna().unique())
    except:
        unique = set(df["rating"].dropna().unique())

    # Binary dataset (1,2)
    if unique <= {1, 2, 1.0, 2.0}:
        df["sentiment"] = df["rating"].apply(
            lambda r: "positive" if str(r).strip() in ("2", "2.0") else "negative"
        )
    else:
        # 1-5 stars
        def map_num(x):
            try:
                x = float(x)
            except:
                return "neutral"
            if x >= 4:
                return "positive"
            if x == 3:
                return "neutral"
            return "negative"

        df["sentiment"] = df["rating"].apply(map_num)

    return df

# ------------------------------
# Clean text
# ------------------------------
def clean_text(text):
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

# ------------------------------
# Main prepare function
# ------------------------------
def load_and_prepare(path=None):
    raw = load_raw_df(path)
    df = normalize_columns(raw)
    df = convert_rating_to_label(df)
    df["review"] = df["review"].fillna("").astype(str)
    df["cleaned"] = df["review"].apply(clean_text)
    df = df.dropna(subset=["cleaned"])
    df = df[df["cleaned"].str.len() > 0]
    return df

if __name__ == "__main__":
    print("Testing preprocess.py...")
    try:
        df = load_and_prepare()
        print(f"✓ Loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Sentiment distribution:\n{df['sentiment'].value_counts()}")
    except Exception as e:
        print(f"✗ Error: {e}")