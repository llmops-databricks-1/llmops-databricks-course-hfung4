# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3 Homework: Simple RAG with Vector Search Implementation
# MAGIC **RAG Flow:**
# MAGIC ```
# MAGIC User Question
# MAGIC     ↓
# MAGIC Vector Search (retrieve relevant documents)
# MAGIC     ↓
# MAGIC Build Prompt (question + context)
# MAGIC     ↓
# MAGIC LLM (generate answer)
# MAGIC     ↓
# MAGIC Response
# MAGIC ```

# COMMAND ----------
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
from loguru import logger

from open_alex_curator.config import load_config, get_env

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup
# MAGIC

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)


# COMMAND ----------
# MAGIC %md
# MAGIC ## RAG with Conversation History
# MAGIC


# COMMAND ----------
class SimpleRAG:
    """Simple RAG system with conversation history."""

    def __init__(self, llm_endpoint: str, index_name: str):
        # Store the Databricks model-serving endpoint name used for LLM calls
        self.llm_endpoint = llm_endpoint
        # Store the fully-qualified vector search index name (<catalog>.<schema>.<index>)
        self.index_name = index_name
        # Maintain a running list of {"role": ..., "content": ...} dicts across turns
        # so the LLM has full conversation context on every call
        self.conversation_history = []

        # Instantiate a Databricks WorkspaceClient using ambient auth (env vars / ~/.databrickscfg).
        # Used only here in __init__ to obtain the host URL and a bearer token for the other clients.
        w = WorkspaceClient()

        # Build an OpenAI-compatible client pointed at the Databricks serving-endpoints gateway.
        # - api_key: strip the "Bearer " prefix from the Authorization header to get the raw PAT/token
        # - base_url: route all requests through the workspace's /serving-endpoints path so the
        #   standard OpenAI SDK can talk to Databricks-hosted models transparently
        self.client = OpenAI(
            api_key=w.config.authenticate()["Authorization"].removeprefix("Bearer "),
            base_url=f"{w.config.host}/serving-endpoints",
        )

        # Build a VectorSearchClient for querying the Databricks Vector Search index.
        # - workspace_url: the Databricks workspace host (e.g. https://adb-<id>.azuredatabricks.net)
        # - personal_access_token: same bearer token stripped of its "Bearer " prefix, used to
        #   authenticate REST calls to the Vector Search service
        self.vsc = VectorSearchClient(
            workspace_url=w.config.host,
            personal_access_token=w.config.authenticate()["Authorization"].removeprefix(
                "Bearer "
            ),
        )

    def retrieve(self, query: str, num_results: int = 5) -> list[dict]:
        """Retrieve relevant documents."""
        # Obtain a handle to the vector search index by its fully-qualified name;
        # this does not fetch any data yet — it just creates a client-side reference
        index = self.vsc.get_index(index_name=self.index_name)

        # Run a hybrid search (dense vector similarity + BM25 keyword) against the index.
        # The SDK embeds query_text automatically using the index's configured embedding model.
        # "hybrid" mode combines semantic and lexical signals for better recall than either alone.
        results = index.similarity_search(
            query_text=query,
            # Columns to return in each result row — order here maps to row[0], row[1], row[2] below
            columns=["text", "title", "open_alex_id"],
            num_results=num_results,  # return at most this many top-k matches
            query_type="hybrid",
        )

        # Accumulate parsed result rows into a list of named dicts
        documents = []
        # Guard against None response or missing "result" key (e.g. empty index or failed call)
        if results and "result" in results:
            # "data_array" is a list of rows; each row is a list of values matching `columns` order
            for row in results["result"].get("data_array", []):
                documents.append(
                    # Map positional row values to named keys for readable downstream access
                    {"text": row[0], "title": row[1], "open_alex_id": row[2]}
                )
        # Return the list of document dicts (empty list if nothing was retrieved)
        return documents

    def chat(self, question: str, num_docs: int = 3) -> str:
        """Chat with LLM, maintaining conversation history"""

        # Step 1: Retrieve documents (a deterministic step).
        # The same question text drives hybrid vector search — no separate embedding call needed.
        documents = self.retrieve(question, num_results=num_docs)

        # Step 2: Concatenate retrieved document texts into a single context block.
        # Each chunk is prefixed with its paper title in brackets so the LLM can
        # attribute claims to specific sources when composing its answer.
        context = "\n\n".join([f"[{doc['title']}]: {doc['text']}" for doc in documents])

        # Step 3: Define the base system instruction that sets the assistant's persona and task.
        system_message = "You are a helpful research assistant. Use the following context from research papers to answer questions."

        # Append the freshly-retrieved context to the system message on every turn.
        # Refreshing the context each turn ensures the grounding material always matches
        # the current question, even as conversation history grows.
        system_message_with_context = f"""{system_message}
CONTEXT:
{context}
If the context doesn't contain relevant information, say so. Always cite paper titles when making claims."""

        # Step 4: Append the user's question to conversation history *before* calling the LLM
        # so that the full history passed in Step 5 already includes this latest turn.
        self.conversation_history.append({"role": "user", "content": question})

        # Step 5: Assemble the full message list for the LLM.
        # The system message (with refreshed context) goes first, followed by all prior turns
        # plus the just-appended user question — giving the model both grounding and memory.
        messages = [
            {"role": "system", "content": system_message_with_context}
        ] + self.conversation_history

        # Step 6: Call the LLM via the OpenAI-compatible Databricks serving endpoint.
        # max_tokens=1000 caps the response length; no temperature set so the model uses its default.
        response = self.client.chat.completions.create(
            model=self.llm_endpoint, messages=messages, max_tokens=1000
        )

        # Extract the assistant's reply text from the first (and only) completion choice.
        # choices is a list because the API supports n>1, but we always use the default n=1.
        answer = response.choices[0].message.content

        # Step 7: Append the assistant's answer to conversation history so it is available
        # as context in subsequent turns, enabling coherent multi-turn dialogue.
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Return the plain text answer to the caller
        return answer

    def clear_history(self):
        """Clear conversation history."""
        # Reset to an empty list, discarding all prior turns.
        # Call this to start a fresh conversation without re-instantiating the object.
        self.conversation_history = []


# COMMAND ----------
# MAGIC %md
# MAGIC ## Test RAG with Conversation History
# MAGIC


# COMMAND ----------
# Create RAG instance
index_name = f"{cfg.project.catalog}.{cfg.project.schema}.open_alex_index"
rag = SimpleRAG(llm_endpoint=cfg.project.llm_endpoint, index_name=index_name)

logger.info("✓ SimpleRAG initialized")

# COMMAND ----------

# Multi-turn conversation
logger.info("Starting multi-turn RAG conversation...")
logger.info("=" * 80)

# First question
question_1 = "What are some examples of LLM for recommender systems?"
answer_1 = rag.chat(question_1)

logger.info(f"Question: {question_1}")
logger.info(f"Answer: {answer_1}\n")


# COMMAND ----------

# Follow-up question (uses conversation history)

question_2 = "What is the performance of these systems?"
answer_2 = rag.chat(question_2)
logger.info(f"Question: {question_2}")
logger.info(f"Answer: {answer_2}\n")

# COMMAND ----------

# Another follow-up question
question_3 = "What are the limitations of these systems?"
answer_3 = rag.chat(question_3)

logger.info(f"Question: {question_3}")
logger.info(f"Answer: {answer_3}")

# COMMAND ----------
