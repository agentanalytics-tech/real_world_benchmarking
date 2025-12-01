import pandas as pd
import logging, os, shutil
import re
from utils import clean_filename_base , rename_files_in_folder
from dotenv import load_dotenv
load_dotenv()   # This loads the .env file

# ==== Config ====
DATA_DIR_SOURCE=os.getenv("DATA_DIR_SOURCE")
os.makedirs(DATA_DIR_SOURCE, exist_ok=True)
LOGS = os.getenv("LOGS")
QUERY_MAP = os.path.join(DATA_DIR_SOURCE,os.getenv("QUERY_MAP_SOURCE_FILE")) 
DATA_DIR_TARGET = os.getenv("DATA_DIR_TARGET")
DATA_DIR_FORMATTED = os.getenv("DATA_DIR_FORMATTED")   
DELIMITER = os.getenv("DELIMITER")

print("query  mamp path",QUERY_MAP)
# ==== Ensure directories exist ====
os.makedirs(LOGS, exist_ok=True)
os.makedirs(DATA_DIR_FORMATTED, exist_ok=True)

# ==== Logging ====
logging.basicConfig(
    filename=os.path.join(LOGS, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_passage_ids(raw,split_by=","):
    """Parses values like:
       [23179372 19270706 23184418]
       "[26997282 21589869 ...]"
       "[123 456\n 789]"
    """
    if pd.isna(raw):
        return []

    text = str(raw)

    # Remove brackets, quotes, and newlines
    text = text.replace("[", "").replace("]", "")
    text = text.replace('"', "").replace("'", "")
    text = text.replace("\n", " ")

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    if text == "":
        return []

    # Split by whitespace — NOT commas
    return text.split(split_by)


def prepare_data(excel_file, passage_folder, passage_id_folder, delimiter="XOXO"):
    df = pd.read_csv(excel_file,encoding="cp1252")
    rename_files_in_folder(passage_folder)
    file_lookup = {os.path.splitext(f)[0]: f for f in os.listdir(passage_folder)}
    print("file lookup",file_lookup)
    prepared = []
    total_matched, total_missing = 0, 0
    for _, row in df.iterrows():
        query_id = row["id"]
        query_text = row["question"]
        relevant_docs = parse_passage_ids(row["doc_ids"])
        relevant_docs = [ clean_filename_base(os.path.splitext(f)[0].lower()) for f in relevant_docs]
        # Add to output CSV
        prepared.append({
            "id": query_id,
            "question": query_text,
            "doc_ids": delimiter.join(relevant_docs)
        })
        
        matched, missing = 0, 0

        # Copy passages
        for doc in relevant_docs:
            if doc in file_lookup:
                f = file_lookup[doc]
                old_path = os.path.join(passage_folder, f)

                base, ext = os.path.splitext(f)
                new_name = f"{base}{delimiter}{query_id}{ext}"
                new_path = os.path.join(passage_id_folder, new_name)

                if not os.path.exists(new_path):
                    shutil.copy2(old_path, new_path)
                    logger.info(f"Copied {f} → {new_name}")

                matched += 1
            else:
                logger.warning(f"No file found for doc_id={doc}")
                missing += 1

        logger.info(f"Finished query_id={query_id}: matched={matched}, missing={missing}")
        total_matched += matched
        total_missing += missing

    # Write CSV
    out_file = os.path.join(LOGS, "prepare_data_output.csv")
    pd.DataFrame(prepared).to_csv(out_file, index=False)

    logger.info(f"Prepared data saved: {out_file}")
    print(f"Prepared data saved: {out_file}")
    print(f"Global Summary: total matched={total_matched}, total missing={total_missing}")


if __name__ == "__main__":
    prepare_data(QUERY_MAP, DATA_DIR_TARGET, DATA_DIR_FORMATTED, delimiter=DELIMITER)
