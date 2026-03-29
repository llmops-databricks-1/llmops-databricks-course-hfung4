# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.1: Custom Functions & Tools for Agents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What are agent tools?
# MAGIC - Creating custom functions
# MAGIC - Tool specifications (OpenAI format)
# MAGIC - Integrating tools with agents
# MAGIC - Vector search as a tool

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Agent Tools
# MAGIC
# MAGIC **Tools** are functions that agents can call to perform specific tasks.
# MAGIC
# MAGIC ### Why Tools?
# MAGIC
# MAGIC LLMs alone cannot:
# MAGIC - Access external data (databases, APIs)
# MAGIC - Perform calculations
# MAGIC - Execute code
# MAGIC - Search documents
# MAGIC
# MAGIC - Tools bridge this gap by giving LLMs the ability to take actions.
# MAGIC
# MAGIC ### Tool Calling Flow:
# MAGIC
# MAGIC ```
# MAGIC User: "What papers discuss transformers?"
# MAGIC   ↓
# MAGIC Agent: Decides to use vector_search tool
# MAGIC   ↓
# MAGIC Tool: vector_search(query="transformers")
# MAGIC   ↓
# MAGIC Tool Result: [paper1, paper2, paper3]
# MAGIC   ↓
# MAGIC Agent: Synthesizes answer from results
# MAGIC   ↓
# MAGIC Response: "Here are papers about transformers..."
# MAGIC ```

# COMMAND ----------
