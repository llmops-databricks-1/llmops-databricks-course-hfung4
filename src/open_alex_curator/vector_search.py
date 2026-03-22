"""Vector search management for OpenAlex papers"""

from databricks.vector_search.client import VectorSearchClient
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
