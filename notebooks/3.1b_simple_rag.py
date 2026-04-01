# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.5: Simple RAG with Vector Search
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is RAG (Retrieval-Augmented Generation)?
# MAGIC - Using Vector Search for document retrieval
# MAGIC - Enriching prompts with retrieved context
# MAGIC - Building a simple Q&A system
# MAGIC
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
# MAGIC ## 1. Setup
# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

cfg.project.llm_endpoint
# COMMAND ----------
# Create clients: Databricks workspace client, OpenAI client,
# and Vector Search client

# Databricks Workspace Client
w = WorkspaceClient()

# OpenAI client for Databricks
client = OpenAI(
    api_key=w.config.authenticate()["Authorization"].removeprefix("Bearer "),
    base_url=f"{w.config.host}/serving-endpoints",
)

# Vector Search client
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.config.authenticate()["Authorization"].removeprefix(
        "Bearer "
    ),
)

logger.info(f"✓ Connected to workspace: {w.config.host}")
logger.info(f"✓ Using LLM endpoint: {cfg.project.llm_endpoint}")
# COMMAND ----------


# MAGIC %md
# MAGIC ## 2. Vector Search Retrieval
# MAGIC
# MAGIC First, let's create a function to retrieve relevant documents from our vector search index.
# MAGIC Note that this retrieval step is done OUTSIDE of the LLM (not decided by LLM),
# MAGIC hence, this is an augmented deterministic LLM workflow and NOT an agentic workflow
# COMMAND ----------
def retrieve_documents(query: str, num_results: int = 5) -> list[dict]:
    """Retrieve relevant documents from vector search.

    Args:
        query: The search query
        num_results: Number of documents to retrieve

    Returns:
        List of document dictionaries with title, text, and metadata
    """
    # Build the fully-qualified index name: <catalog>.<schema>.<index> — matches how it was created
    index_name = f"{cfg.project.catalog}.{cfg.project.schema}.open_alex_index"
    # Get a handle to the vector search index (does not fetch data yet)
    index = vsc.get_index(index_name=index_name)

    # Run a hybrid search (combines dense vector similarity + keyword BM25) against the index
    results = index.similarity_search(
        query_text=query,  # the user's natural-language query; the SDK embeds this automatically
        # specify which columns to return in results — order here determines row[i] positions below
        columns=["text", "title", "open_alex_id", "authors", "year"],
        num_results=num_results,  # how many top-k results to return
        query_type="hybrid",  # "hybrid" = dense + sparse; use "ann" for pure vector search
    )

    # Accumulate parsed documents into a list to return
    documents = []
    # Guard: results can be None or missing "result" key if the index is empty or the call failed
    if results and "result" in results:
        # "data_array" is a list of rows; each row is a list of values in the same order as `columns`
        data_array = results["result"].get("data_array", [])
        # Convert each raw row (list) into a named dict for easier downstream access
        for row in data_array:
            documents.append(
                {
                    "text": row[
                        0
                    ],  # full text of the paper chunk (used as RAG context)
                    "title": row[1],  # paper title
                    "open_alex_id": row[2],  # unique OpenAlex identifier for the paper
                    "authors": row[3],  # author list
                    "year": row[4],  # publication year
                }
            )
    # Return the list of document dicts (empty list if no results found)
    return documents


# COMMAND ----------
# Test retrieval
query = "transformer attention mechanisms"
docs = retrieve_documents(query, num_results=3)

logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")

for i, doc in enumerate(docs, 1):
    logger.info(f"\n{i}. {doc['title']}")
    logger.info(f"   Open Alex ID: {doc['open_alex_id']}")
    logger.info(f"   Text preview: {doc['text'][:150]}...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Building the RAG Prompt
# MAGIC
# MAGIC Now let's create a function that builds a prompt with the retrieved context.
# COMMAND ----------


def build_rag_prompt(question: str, documents: list[dict]) -> str:
    """Build a prompt with retrieved context

    Args:
        question: The user's question
        documents: List of retrieved documents

    Returns:
        Formatted prompt string
    """

    # Format each retrieved document into a labeled text block.
    # enumerate(documents, 1) gives 1-based numbering for human-readable labels.
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(
            f"Document {i}: {doc['title']}\n"
            f"OpenAlex ID: {doc['open_alex_id']}\n"
            f"Context: {doc['text']}"
        )

    # Join document blocks with a horizontal-rule separator so the LLM sees clear
    # boundaries between individual documents.
    context = "\n---\n".join(context_parts)

    # Prompt structure follows: system instruction → retrieved context → user question
    # → output cue ("ANSWER:").  Explicit section headers (CONTEXT / QUESTION /
    # INSTRUCTIONS) help most LLMs parse the prompt reliably.
    prompt = f"""You are a helpful research assistant. Answer the question based on the provided context from research papers.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer based on the provided context
- If the context doesn't contain enough information, say so
- Cite the relevant paper titles when making claims
- Be concise but thorough

ANSWER:"""

    # Return the fully assembled prompt string to the caller, which will pass it
    # to the LLM as the user message (or a combined system+user message).
    return prompt


# COMMAND ----------

# Test prompt building
query = "What is LLM4Rec"
docs = retrieve_documents(query, num_results=3)

logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")

for i, doc in enumerate(docs, 1):
    logger.info(f"\n{i}. {doc['title']}")
    logger.info(f"   Open Alex ID: {doc['open_alex_id']}")
    logger.info(f"   Text preview: {doc['text'][:150]}...")


test_prompt = build_rag_prompt(query, docs)
logger.info("Built RAG prompt:")
logger.info(f"Prompt length: {len(test_prompt)} characters")
logger.info(f"Preview:\n{test_prompt[:500]}...")

# COMMAND ----------


# MAGIC %md
# MAGIC ## 4. RAG Query Function
# MAGIC
# MAGIC Combine retrieval and generation into a single function.
# COMMAND ----------
def rag_query(question: str, num_docs: int = 5) -> dict:
    """Answer a question using RAG.

    Args:
        question: The user's question
        num_docs: Number of documents to retrieve

    Returns:
        Dictionary with answer and sources
    """

    # Step 1: Retrieve relevant documents from the vector search index.
    # The same question text is used both for retrieval (dense/hybrid search)
    # and later for the LLM — no separate embedding step is needed here.
    logger.info(f"Retrieving documents for: '{question}'")
    documents = retrieve_documents(question, num_results=num_docs)
    logger.info(f"Retrieved {len(documents)} documents")

    # Step 2: Inject the retrieved documents into a structured prompt.
    # build_rag_prompt formats them as labeled context blocks so the LLM
    # knows exactly which source each passage came from.
    prompt = build_rag_prompt(question, documents)

    # Step 3: Send the enriched prompt to the LLM and get a grounded answer.
    logger.info("Generating answer...")
    response = client.chat.completions.create(
        model=cfg.project.llm_endpoint,  # Databricks serving endpoint name from project config
        # Single "user" turn: the entire RAG prompt (system instruction + context + question)
        # is passed as a user message, matching how Databricks-hosted models expect input.
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,  # cap response length; 1000 tokens is enough for most research summaries
        # 0.0 forces deterministic, maximally grounded output, higher temp enforces more creative responses
        temperature=0.7,
    )

    # Extract the assistant's reply text from the first (and only) completion choice.
    # response.choices is a list because the API supports n>1 completions, but we
    # always request the default n=1, so index 0 is safe here.
    answer = response.choices[0].message.content

    # Return a structured dict containing the original question, the LLM's answer,
    # and a  list of source citations (title + OpenAlex ID) so callers
    # can display provenance alongside the answer.
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"title": doc["title"], "open_alex_id": doc["open_alex_id"]}
            for doc in documents
        ],
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test RAG System

# COMMAND ----------

# Test with a research question
result_1 = rag_query("What are the key innovations in transformer architectures?")

logger.info("=" * 80)
logger.info(f"Question: {result_1['question']}")

logger.info("=" * 80)
logger.info(f"\nAnswer:\n{result_1['answer']}")

logger.info("\nSources:")
for source in result_1["sources"]:
    logger.info(f"  -{source['title']} ({source['open_alex_id']})")
# COMMAND ----------
# Test with another research question
result_2 = rag_query("How do large language models handle reasoning tasks?")

logger.info("=" * 80)
logger.info(f"Question: {result_2['question']}")
logger.info("=" * 80)
logger.info(f"\nAnswer:\n{result_2['answer']}")
logger.info("\nSources:")
for source in result_2["sources"]:
    logger.info(f"  - {source['title']} ({source['open_alex_id']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. RAG with Conversation History
# MAGIC
# MAGIC Extend RAG to support multi-turn conversations.


# COMMAND ----------
class SimpleRAG:
    """Simple RAG system with conversation history."""

    def __init__(self, llm_endpoint: str, index_name: str):
        self.llm_endpoint = llm_endpoint
        self.index_name = index_name
        self.conversation_history = []

        # Initialize clients

        # Databricks Workspace Client
        w = WorkspaceClient()

        # OpenAI client for Databricks
        self.client = OpenAI(
            api_key=w.config.authenticate()["Authorization"].removeprefix("Bearer "),
            base_url=f"{w.config.host}/serving-endpoints",
        )

        # Vector Search client
        self.vsc = VectorSearchClient(
            workspace_url=w.config.host,
            personal_access_token=w.config.authenticate()["Authorization"].removeprefix(
                "Bearer "
            ),
        )

    def retrieve(self, query: str, num_results: int = 5) -> list[dict]:
        """Retrieve relevant documents."""
        index = self.vsc.get_index(index_name=self.index_name)
        results = index.similarity_search(
            query_text=query,
            # Columns to return in the retrieved results
            columns=["text", "title", "open_alex_id"],
            num_results=num_results,
            query_type="hybrid",
        )

        documents = []
        if results and "result" in results:
            for row in results["result"].get("data_array", []):
                documents.append(
                    {"text": row[0], "title": row[1], "open_alex_id": row[2]}
                )
        return documents

    def chat(self, question: str, num_docs: int = 3) -> str:
        """Chat with LLM, maintaining conversation history"""

        # Step 1: Retrieve documents (a deterministic step)
        documents = self.retrieve(question, num_results=num_docs)

        # Step 2: Build context
        context = "\n\n".join([f"[{doc['title']}]: {doc['text']}" for doc in documents])

        # Step 3: Build system message with context
        system_message = "You are a helpful research assistant. Use the following context from research papers to answer questions."

        system_message_with_context = f"""{system_message}
CONTEXT:
{context}
If the context doesn't contain relevant information, say so. Always cite paper titles when making claims."""

        # Step 4: Add user message (question) to conversation history (becoming its most recent entry)
        # before generating responses with LLM
        self.conversation_history.append({"role": "user", "content": question})

        # Step 5: Build full message for LLM (system message with context AND conversation history (that contains newest question from user)
        messages = [
            {"role": "system", "content": system_message_with_context}
        ] + self.conversation_history

        # Step 6: Generate response
        response = self.client.chat.completions.create(
            model=self.llm_endpoint, messages=messages, max_tokens=1000
        )

        answer = response.choices[0].message.content

        # Step 7: Log assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": answer})

        return answer

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
