"""Vector search management for OpenAlex papers"""

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
from loguru import logger

from open_alex_curator.config import Config


class VectorSearchManager:
    """Manages vector search endpoints and indexes for OpenAlex paper chunks"""

    def __init__(
        self,
        config: Config,
        endpoint_name: str | None = None,
        embedding_model: str | None = None,
        usage_policy_id: str | None = None,
    ) -> None:
        """Initialize VectorSearchManager.

        Args:
            config: Config object
            endpoint_name: Name of the vector search endpoint (uses config if None)
            embedding_model: Name of the embedding model endpoint (uses config if None)
            usage_policy_id: Optional Databricks usage policy ID to attach to the
                index, for governance/cost tracking purposes.
        """
        self.config = config
        self.endpoint_name = endpoint_name or config.project.vector_search_endpoint
        self.embedding_model = embedding_model or config.embedding_endpoint
        self.catalog = config.catalog
        self.schema = config.schema
        # Optional — only required if the workspace enforces a usage policy on indexes
        self.usage_policy_id = usage_policy_id

        # Databricks SDK client for managing vector search endpoints and indexes
        self.client = VectorSearchClient()
        # Full Unity Catalog path: <catalog>.<schema>.<index_name>
        self.index_name = f"{self.catalog}.{self.schema}.open_alex_index"

    def create_endpoint_if_not_exists(self) -> None:
        """Create vector search endpoint if it doesn't exist."""
        # list_endpoints() can return either a dict with an "endpoints" key
        # or an SDK object — handle both to extract the list safely
        endpoint_response = self.client.list_endpoints()
        endpoints = (
            endpoint_response.get("endpoints", [])
            if isinstance(endpoint_response, dict)
            else []
        )

        # Check if an endpoint with our target name already exists.
        # Each element can be a dict or an SDK object, so check both forms.
        endpoint_exists = any(
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None))
            == self.endpoint_name
            for ep in endpoints
        )

        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {self.endpoint_name}")
            # create_endpoint_and_wait blocks until the endpoint is ONLINE,
            # so no additional polling is needed after this call.
            # endpoint_type="STANDARD" provisions a dedicated serving cluster.
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name,
                endpoint_type="STANDARD",
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search endpoint created: {self.endpoint_name}")
        else:
            # Endpoint already exists — nothing to do
            logger.info(f"✓ Vector search endpoint already exist: {self.endpoint_name}")

    def create_or_get_index(self) -> VectorSearchIndex:
        """Create or get vector search index.

        Returns:
            Vector search index object
        """
        # Ensure the vector search endpoint is running before creating an index on it
        self.create_endpoint_if_not_exists()

        # Source table containing cleaned chunk text produced by process_chunks().
        # The index embeds the "text" column and stays in sync with this Delta table.
        source_table = f"{self.catalog}.{self.schema}.open_alex_chunks_table"

        # Try to fetch an existing index first to keep this method idempotent —
        # safe to call on every pipeline run without recreating the index each time.
        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return index
        except Exception:
            # get_index raises if the index doesn't exist yet; fall through to create it
            logger.info(f"Index {self.index_name} not found, it will now be created...")

        # Create a Delta Sync index backed by the source table.
        # pipeline_type="TRIGGERED" means embeddings are synced on demand, not streaming.
        # primary_key="id" uniquely identifies each chunk row.
        # embedding_source_column="text" is the column whose content gets embedded.
        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search index created: {self.index_name}")
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            # Race condition: index was created between get_index and create calls.
            # Retry get_index to return the already-existing index.
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return self.client.get_index(index_name=self.index_name)

    def sync_index(self) -> None:
        """Sync the vector search index with the source table."""
        # Get or create index
        index = self.create_or_get_index()
        logger.info(f"Syncing vector search index: {self.index_name}")
        # Sync the vector search index
        index.sync()
        logger.info("✓ Index sync triggered")

    def search(
        self,
        query: str,
        num_results: int = 5,
        filters: dict | None = None,
    ) -> dict:
        """Search the vector index.

        Args:
            query: Search query text
            num_results: Number of results to return
            filters: Optional filters to apply

        Returns:
            Search results dictionary
        """
        # Get index
        index = self.client.get_index(index_name=self.index_name)
        # Query against the index
        results = index.similarity_search(
            query_text=query,
            # columns to return along with each of the search result
            columns=[
                "id",
                "text",
                "open_alex_id",
                "title",
                "authors",
                "summary",
                "year",
                "month",
                "day",
            ],
            num_results=num_results,
            filters=filters,
        )
        return results
