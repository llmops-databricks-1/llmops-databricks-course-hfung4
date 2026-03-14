# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 1.2: Provisioned Throughput Deployment
# MAGIC - Your own fine-tuned models
# MAGIC - Custom models registered in Unity Catalog
# MAGIC - Models that need dedicated capacity
# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    AiGatewayUsageTrackingConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)
import time
from loguru import logger
from openai import OpenAI

w = WorkspaceClient()
# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Provisioned Throughput
# MAGIC
# MAGIC ### Model Units and Throughput
# MAGIC
# MAGIC - **Model Unit**: A unit of compute capacity for serving models
# MAGIC - **Throughput**: Measured in tokens per second
# MAGIC - **Example**: For DeepSeek/Llama models:
# MAGIC   - 1 model unit ≈ 65 tokens/second
# MAGIC   - 50 model units ≈ 3,250 tokens/second

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Advanced Configuration Parameters
# MAGIC
# MAGIC ### Available Parameters for Provisioned Throughput:
# MAGIC
# MAGIC #### Core Parameters:
# MAGIC - **workload_size**: `Small`, `Medium`, `Large` - Compute capacity per instance
# MAGIC - **scale_to_zero_enabled**: `True`/`False` - Auto-scale to zero when idle
# MAGIC - **min_provisioned_throughput**: Minimum model units (must be 0 if scale_to_zero enabled)
# MAGIC - **max_provisioned_throughput**: Maximum model units for auto-scaling
# MAGIC
# MAGIC #### Monitoring & Observability:
# MAGIC - **Inference Tables**: Log all requests/responses to Delta table
# MAGIC   - Must be enabled via Databricks UI (Serving → Endpoint → Configuration)
# MAGIC   - Or via REST API (not available in SDK 0.78.0)
# MAGIC   - Creates table: `{catalog}.{schema}.{endpoint_name}_payload`
# MAGIC   - Includes: request_id, timestamp, request, response, status_code, latency
# MAGIC   - Useful for: debugging, auditing, model monitoring, fine-tuning data
# MAGIC
# MAGIC #### Safety & Compliance:
# MAGIC - **guardrails**: Configure input/output validation and filtering
# MAGIC   - PII detection and redaction
# MAGIC
# MAGIC #### Environment Variables:
# MAGIC - **environment_vars**: Pass custom environment variables to the model
# MAGIC   - API keys for external services
# MAGIC   - Custom configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Deploying LLama with Provisioned Throughput

# COMMAND ----------
