"""Vector search management for OpenAlex papers"""

from databricks.vector_search.client import VectorSearchClient

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
