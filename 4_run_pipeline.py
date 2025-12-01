import os
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from functools import partial
import concurrent.futures

# Your clients / utils
from waveflowdb_client import VectorLakeClient
from utils import convert_to_sql_vql
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

# ======================
# CONFIG
# ======================
RESULTS_DIR = os.getenv("RESULTS_DIR")
print("result dir",RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)
query_log_file = os.path.join(os.getenv("DATA_DIR_SOURCE"), os.getenv("QUERY_MAP_SOURCE_FILE"))
print("qery map",query_log_file)
DELIMITER = os.getenv("DELIMITER")
# Multiprocessing workers
NUM_WORKERS = int(os.getenv("MAX_WORKERS_QUERY"))
# ======================
# Lazy globals (per-process)
# ======================
_model = None
_index = None
_waveflow_client = None


def get_model():
    """Lazy-load embedding model per-process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(os.getenv("MODEL_NAME"))
    return _model


def get_pinecone_index():
    """Lazy-init Pinecone Index per-process."""
    global _index
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    return _index


def get_waveflow_client():
    """Lazy-init Waveflow (VectorLake) client per-process."""
    global _waveflow_client
    if _waveflow_client is None:
        _waveflow_client = VectorLakeClient(api_key=os.getenv("WAVEFLOWDB_API_KEY"),
                                             host=os.getenv("BASE_URL"))
    return _waveflow_client

# ======================
# LOGGING
# ======================
logging.basicConfig(
    filename=os.path.join(RESULTS_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Also print to stdout for convenience
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

# ======================
# METRICS
# ======================

def calculate_mrr(relevant_docs, retrieved_docs_list):
    relevant_set = set(relevant_docs)
    for rank, doc_id in enumerate(retrieved_docs_list, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def calculate_ndcg(relevant_docs, retrieved_docs_list, k=10):
    if not relevant_docs:
        return 0.0
    relevant_set = set(relevant_docs)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs_list[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

# ======================
# Query functions (multiprocessing-safe)
# ======================

def query_pinecone(query_text, top_k, HYBRID_FILTER=False, deduplicate=True):
    """
    Multiprocessing-safe Pinecone query. Uses get_model() and get_pinecone_index()
    to ensure per-process initialization.
    Returns (latency_dict, stats_dict, docs_list, status)
    """
    model = get_model()
    index = get_pinecone_index()

    stats = {
        "original_query": query_text,
        "final_query": query_text,  # Pinecone does NOT transform queries
        "timestamp": datetime.now().isoformat(),
    }
    latency = {}
    status = "success"

    try:
        embed_start = time.time()
        embedding = model.encode(query_text).tolist()
        latency["embedding_time"] = time.time() - embed_start

        fetch_k = top_k * 2 if deduplicate else top_k

        q_start = time.time()
        results = index.query(vector=embedding, top_k=fetch_k, include_metadata=True)
        latency["query_time"] = time.time() - q_start
        latency["total_time"] = latency["embedding_time"] + latency["query_time"]

        response_dict = results.to_dict() if hasattr(results, "to_dict") else results
        raw_docs = [m.get("id") for m in response_dict.get("matches", []) if m.get("id")]

        # deduplicate by base id (split by XOXO)
        if deduplicate:
            seen = set()
            unique = []
            for d in raw_docs:
                base = d.split("XOXO")[0] if "XOXO" in d else d
                if base not in seen:
                    seen.add(base)
                    unique.append(base)
            docs = unique[:top_k]
        else:
            docs = raw_docs[:top_k]

    except Exception:
        logger.exception("Pinecone query error")
        docs = []
        status = "error"
        latency.setdefault("embedding_time", 0.0)
        latency.setdefault("query_time", 0.0)
        latency["total_time"] = latency.get("embedding_time", 0.0) + latency.get("query_time", 0.0)

    return latency, stats, docs, status


def query_waveflow(query_text, top_k, HYBRID_FILTER=False, deduplicate=True):
    """
    Multiprocessing-safe Waveflow (VectorLake) query. Uses get_waveflow_client().
    Returns (latency_dict, stats_dict, docs_list, status)
    """
    client = get_waveflow_client()

    final_query = convert_to_sql_vql(query_text,type=os.getenv("TYPE")) if HYBRID_FILTER else query_text

    stats = {"original_query": query_text, "final_query": final_query, "timestamp": datetime.now().isoformat()}
    latency = {}
    status = "success"

    try:
        fetch_k = top_k * 2 if deduplicate else top_k
        q_start = time.time()
        logging.info(f"Query asked: {final_query}")
        
        results = client.get_matching_docs(
            query=final_query,
            session_id=os.getenv("SESSION_ID"),
            user_id=os.getenv("USER_ID"),
            vector_lake_description=os.getenv("NAMESPACE"),
            with_data=True,
            hybrid_filter=HYBRID_FILTER,
            top_docs=fetch_k,
            threshold=0.15
         )

        # Expect results['reply']['content'] to be a list of dicts with 'file_name'
        # print("results",results)
        raw_docs = []
        try:
            raw_docs = [d.get("file_name") for d in results.get("reply", {}).get("content", []) if d.get("file_name")]
        except Exception:
            raw_docs = []

        logging.info(f"raw docs {raw_docs}")

        # Normalize and deduplicate
        if deduplicate:
            seen = set()
            unique = []
            for d in raw_docs:
                if not d:
                    continue
                base = d.split("xoxo")[0] if "xoxo" in d else d
                if base not in seen:
                    seen.add(base)
                    unique.append(base)
            docs = unique[:top_k]
        else:
            docs = raw_docs[:top_k]

        latency["embedding_time"] = 0.0
        latency["query_time"] = time.time() - q_start
        latency["total_time"] = latency["query_time"]

    except Exception as e:
        logger.exception(f"{final_query}Waveflow query error {e}")
        docs = []
        status = "error"
        latency["embedding_time"] = 0.0
        latency["query_time"] = 0.0
        latency["total_time"] = 0.0

    return latency, stats, docs, status

# ======================
# Worker / Evaluator (updated: includes status + result_text trimmed)
# ======================
# ======================

def _run_single_query(row_dict, query_func, top_k, HYBRID_FILTER):
    """
    row_dict: plain dict with keys 'query_id', 'query_text', 'relevant_docs' (string)
    query_func: one of query_pinecone or query_waveflow
    """
    query_id = row_dict.get("query_id")
    query_text = row_dict.get("query_text", "")

    relevant_docs = [
        os.path.splitext(doc.strip())[0].lower()
        for doc in str(row_dict.get("relevant_docs", "")).split(",")
        if doc and doc.strip()
    ]

    # Run the query
    latency, stats, retrieved_docs, status = query_func(query_text, top_k, HYBRID_FILTER=HYBRID_FILTER)
    retrieved_docs = [s.lower() for s in retrieved_docs]

    hits = sum(1 for d in retrieved_docs if d in relevant_docs)
    precision = hits / len(retrieved_docs) if retrieved_docs else 0.0
    recall = hits / len(relevant_docs) if relevant_docs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mrr = calculate_mrr(relevant_docs, retrieved_docs)
    ndcg = calculate_ndcg(relevant_docs, retrieved_docs, k=top_k)

    result = {
        "query_id": query_id,
        "query_text": query_text,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "ndcg": ndcg,
        "embedding_time": latency.get("embedding_time", 0.0),
        "query_time": latency.get("query_time", 0.0),
        "total_time": latency.get("total_time", 0.0),
        "retrieved_docs": ", ".join(retrieved_docs)[:300],
        "relevant_docs": ", ".join(relevant_docs)[:300],
        "status": "success" if retrieved_docs else "no_results",
        "raw_result": str(retrieved_docs)[:500]
    }

    # Small log for each query (keeps stdout light)
    logger.info(
        f"[{query_func.__name__} | Query {query_id}] "
        f"Status={status} | FinalQuery='{stats.get('final_query')}' | "
        f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, "
        f"MRR={mrr:.3f}, nDCG@{top_k}={ndcg:.3f} | "
        f"Time={result['total_time']:.3f}s"
    )

    return result


def evaluate_queries_parallel(df, query_func, system_name, top_k=os.getenv("TOP_K")
, HYBRID_FILTER=False, num_workers=NUM_WORKERS):
    """
    Evaluate all rows in df in parallel using ProcessPoolExecutor.
    Returns list of result dicts (one per query).
    """
    logger.info(f"🔎 Running {system_name} in parallel with {len(df)} queries (workers={num_workers})...")

    # Convert rows to plain dicts (safer for pickling)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "query_id": row.get("id"),
            "query_text": row.get("question", ""),
            "relevant_docs": row.get("doc_ids", ""),
        })
    worker = partial(_run_single_query, query_func=query_func, top_k=top_k, HYBRID_FILTER=HYBRID_FILTER)

    results = []
    # Use ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # map preserves order — we can also use as_completed if we want streaming
        for res in executor.map(worker, rows):
            # attach system metadata at the end to keep worker returns simple
            res["system"] = system_name
            results.append(res)
# Persist results CSV
    out_file = os.path.join(RESULTS_DIR, f"{system_name}_results_top{top_k}_hybrid{HYBRID_FILTER}.csv")
    try:
        pd.DataFrame(results).to_csv(out_file, index=False)
        logger.info(f"📄 {system_name} results written to {out_file}")
    except Exception:
        logger.exception("Failed to write per-system CSV")

    return results

# ======================
# Main driver
# ======================

def main():
    df = pd.read_csv(query_log_file)

    TOP_K_LIST = [2, 5, 10]          # all k values combined
    HYBRID_FILTER_LIST = [True, False] # both filter settings

    all_results = []

    for TOP_K in TOP_K_LIST:
        for HYBRID_FILTER in HYBRID_FILTER_LIST:
            pinecone_res = []
            # Waveflow always evaluated
            waveflow_res = evaluate_queries_parallel(
                df, query_waveflow, "waveflow", top_k=TOP_K, HYBRID_FILTER=HYBRID_FILTER, num_workers=NUM_WORKERS
            )
            for r in waveflow_res:
                r["top_k"] = TOP_K
                r["hybrid_filter"] = HYBRID_FILTER

            if not HYBRID_FILTER:
                pinecone_res = evaluate_queries_parallel(
                    df, query_pinecone, "pinecone", top_k=TOP_K, HYBRID_FILTER=HYBRID_FILTER, num_workers=NUM_WORKERS
                )
                for r in pinecone_res:
                    r["top_k"] = TOP_K
                    r["hybrid_filter"] = HYBRID_FILTER

            # Collect all results
            all_results.extend(pinecone_res + waveflow_res)

            # Write merged CSV for this setting
            merged = pd.DataFrame(pinecone_res + waveflow_res)
            merged_file = os.path.join(
                RESULTS_DIR, f"merged_results_top{TOP_K}_hybrid{HYBRID_FILTER}.csv"
            )
            try:
                merged.to_csv(merged_file, index=False)
                logger.info(f"📊 Merged results written to {merged_file}")
            except Exception:
                logger.exception("Failed to write merged CSV for setting")

    # --- Final aggregation ---
    final_df = pd.DataFrame(all_results)

    if not final_df.empty:
        agg_df = final_df.groupby(["system", "top_k", "hybrid_filter"]).agg(
            avg_precision=("precision", "mean"),
            avg_recall=("recall", "mean"),
            avg_f1=("f1", "mean"),
            avg_mrr=("mrr", "mean"),
            avg_ndcg=("ndcg", "mean")
            # avg_embedding_time=("embedding_time", "mean"),
            # avg_query_time=("query_time", "mean"),
            # avg_total_time=("total_time", "mean"),
        ).reset_index()
    else:
        agg_df = pd.DataFrame()

    # Save to Excel (2 sheets: raw + aggregated)
    final_file = os.path.join(RESULTS_DIR, "all_results.xlsx")
    try:
        with pd.ExcelWriter(final_file, engine="openpyxl") as writer:
            final_df.to_excel(writer, sheet_name="raw_results", index=False)
            agg_df.to_excel(writer, sheet_name="aggregated", index=False)
        logger.info(f"📊 Results written to {final_file} (raw + aggregated)")
    except Exception:
        logger.exception("Failed to write final Excel file")


if __name__ == "__main__":
    main()
