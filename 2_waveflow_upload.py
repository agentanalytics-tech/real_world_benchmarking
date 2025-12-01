
from waveflowdb_client import VectorLakeClient, Config
from waveflowdb_client import utils
from dotenv import load_dotenv
import os
import logging
session_id = "session"
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================
# Client Initialization
# ======================
client = VectorLakeClient(
    api_key=os.getenv("WAVEFLOWDB_API_KEY"),
    host=os.getenv("BASE_URL")  ,
    vector_lake_path=os.getenv("DATA_DIR_FORMATTED"),
    # service_port=8030
    )


# ======================
# Upload data
# ======================

results = client.add_documents(
    user_id=os.getenv("USER_ID"),
    vector_lake_description=os.getenv("NAMESPACE")
)
logger.info(results)
