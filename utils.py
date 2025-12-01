import spacy
import csv
import os
import re
import unicodedata
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------
# Load SpaCy models
# --------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# --------------------------------------------------
# STOPLIST — remove ONLY useless words
# --------------------------------------------------
STOPWORDS = {
    # question words
    "what", "which", "why", "how", "where", "who", "when",
    # intent verbs
    "describe", "list", "show", "explain", "define", "summarize", "share",
    "give", "tell", "provide", "information", "info", "overview", "details",
    # auxiliaries
    "is", "are", "be", "been", "being", "am", "was", "were",
    # generic filler
    "type", "types", "kind", "kinds", "thing", "things",
    # prepositions / determiners
    "a","an","the","of","for","to","from","by","with",
    "in","on","at","as"
}


# --------------------------------------------------
# Keep word content but strip tail punctuation
# Keep / and -
# --------------------------------------------------
def clean_token(tok: str):
    return re.sub(r"[^\w\-/]+$", "", tok).strip()


# --------------------------------------------------
# Extract & Rank:  PROPN > NOUN > BIOMED
# --------------------------------------------------
def extract_keywords(text: str):
    doc = nlp(text)

    proper = []
    nouns = []
    biomed = []

    # POS-based
    for tok in doc:
        w = clean_token(tok.text)
        if not w:
            continue

        if w.lower() in STOPWORDS:
            continue

        if tok.pos_ == "PROPN":
            proper.append(w)
        elif tok.pos_ == "NOUN":
            nouns.append(w)

    # Dedupe while preserving ranking priority
    seen = set()
    ranked = []
    for lst in [proper, nouns, biomed]:
        for w in lst:
            if w not in seen:
                seen.add(w)
                ranked.append(w)

    return ranked


# --------------------------------------------------
# Main VQL Builder
# --------------------------------------------------
def convert_to_sql_vql(query: str, type):
    keywords = extract_keywords(query)
    print("keyword", keywords)
    # Cap = 6 keywords total
    keywords = keywords[:6]

    # Create pairs of 2 → braces
    groups = []
    for i in range(0, len(keywords), 2):
        pair = keywords[i:i+2]
        
        if len(pair) == 2:
            # Pair of 2 keywords: {keyword1 type keyword2}
            groups.append(f"{{{pair[0]} {type} {pair[1]}}}")
        elif len(pair) == 1:
            # Odd/Leftover keyword: {keyword1}
            groups.append(f"{{{pair[0]}}}")
        # If len(pair) is 0, nothing is added to groups

    # Up to max 3 braces
    groups = groups[:3]

    # Build VQL
    vql = f'SELECT TOP 10 WHERE query is "{query}"'
    if groups:
        vql += " CONTAINS " + " AND ".join(groups)

    return vql

# Windows reserved filenames
WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10))
}


def clean_filename_base(fname: str, max_len: int = 100) -> str:
    """
    Cleans ONLY filename base.
    DOES NOT alter extension.
    Safe ASCII filename.
    """
    name = str(fname).strip()

    # Unicode → ASCII
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    name = name.lower()

    # Replace spaces/tabs with underscore
    name = re.sub(r"\s+", "_", name)

    # Remove invalid filesystem chars
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', name)

    # Replace unknown groups with underscore
    name = re.sub(r'[^a-z0-9._-]+', '_', name)

    # Collapse underscores
    name = re.sub(r'_+', '_', name)
    name = name.strip("._- ")

    # If empty after cleaning
    if not name:
        name = "unnamed_file"

    # Reserved Windows names
    if name in WINDOWS_RESERVED:
        name = f"{name}_file"

    return name[:max_len].lower()


def rename_files_in_folder(folder_path: str):
    """
    Renames all files in the folder using clean_filename_base + lowercase.
    Handles conflicts and case-only renames safely.
    """
    try:
        if not os.path.isdir(folder_path):
            print(f"Error: Folder not found at '{folder_path}'")
            return

        print(f"Starting to rename files in: {folder_path}\n")

        for filename in os.listdir(folder_path):
            old_path = os.path.join(folder_path, filename)

            # Skip directories
            if os.path.isdir(old_path):
                continue

            name, ext = os.path.splitext(filename)

            # Clean
            clean_name = clean_filename_base(name).lower()
            new_filename = clean_name + ext.lower()  # ext lower optional
            new_path = os.path.join(folder_path, new_filename)

            # If identical, skip
            if filename == new_filename:
                continue

            # If target exists but is *exact same file* but different case
            # Example: Test.txt → test.txt
            if os.path.exists(new_path):
                # Windows case: rename in two steps
                temp_path = os.path.join(folder_path, f"__tmp__{new_filename}")
                os.rename(old_path, temp_path)
                os.rename(temp_path, new_path)
                print(f"[case-fix] '{filename}' -> '{new_filename}'")
            else:
                os.rename(old_path, new_path)
                print(f"Renamed: '{filename}' -> '{new_filename}'")

        print("\nFile renaming process completed.")
    except Exception as e:
        print(f"Unexpected error: {e}")
