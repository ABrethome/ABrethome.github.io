---
title: "Create and consume a MCP server with observations (Part 1)"
date: 2025-08-23
permalink: /posts/2025/08/mcp-servers-with-observations-part-1/
tags:
  - GenAI
  - MCP
  - Langchain
---

*This post is for data scientists, AI engineers, and researchers interested in building trustworthy, reproducible AI systems with LangChain and MCP. Familiarity with Python and basic concepts of LLMs and RAG is recommended.*
{: .notice--info}

**Note:** All code is available in my Github repository called [Sandbox](https://github.com/ABrethome/Sandbox).
{: .notice}

## Table of Contents
1. [Introduction](#introduction)
2. [What are Observations?](#observations-what-are-those)
3. [The Objective](#the-objective)
4. [Creating a Fake Database](#lets-create-a-fake-database)
5. [Building the MCP Server](#lets-create-the-mcp-server)
6. [Testing with a ReACT Agent](#test-the-mcp-server-with-a-basic-react-agent)
7. [Summary & Next Steps](#summary--next-steps)

## Introduction

In the evolving world of Model Context Protocol (MCP) servers and AI agents, how we handle sources is more important than ever, especially in scientific and enterprise settings. Most AI tools today, like Microsoft Copilot 365, tend to provide sources as simple URLs. But as any researcher knows, not all valuable information lives on a web page. Critical data often hides in SQL databases, research documents, images, videos, or audio files, and we need to be able to source all of these.

For scientists, trust and reproducibility are non-negotiable. It's not enough to just get an answer; I need to know exactly where that answer came from and how it was derived. If a language model (LLM) passes source material through itself, there's a risk of hallucination or accidental changes, making it impossible to fully trust the source. Instead, sources should be referenced and cached, with the LLM providing only a pointer to the source, never the content itself. This way, I can independently verify every source for authenticity and existence, and I never have to worry about made-up references.

At the end of the day, scientific rigor is all about transparency and traceability. An answer is only as good as my ability to reproduce it using the sources provided. To truly support scientific work, MCP servers must handle a broad spectrum of source types, not just URLs, but also SQL queries, documents, and multimedia files. As a scientist, I want to be empowered with the trust and visibility I need to do my best work.

> **Real-world use case:**
> In regulated industries or academic research, being able to trace every answer back to its original, immutable source is essential for compliance, audit trails, and reproducibility.
{: .notice--success}

Have you ever wanted to build a smart MCP server that not only answers questions, but also tells you where those answers come from with 100% certainty? In this post series, I'll walk you through creating an MCP (Model Context Protocol) server, along with a simple ReACT agent that can ask questions and get answers (Part 1). Afterward, we'll build a more advanced agent that can call the MCP server and provide answers with the sources (observations) behind them, without any hallucinated sources. Whether you're new to MCP or just curious about how to connect answers with their evidence, I hope you'll find this guide approachable and practical. Let's dive in!

## Observations, what are those?

Here, Observations refer to any type of data retrieved by an agent to support an answer. The most obvious type of observation is RAG, where observations are chunks of documents used to provide an answer by the LLM. But there are lots of different observations, so we will remain generic where an observation is defined as below:

```python
from pydantic import BaseModel, Field
from typing import Any
class Observation(BaseModel):
    """Describe an observation used to ground an answer."""
    id: str | None = None
    type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- `id`: Unique identifier of the Observation, e.g. a UUID. This is the pointer the LLM can use to reference an observation, instead of using a URL.
- `type`: The different types of observations such as `document`, `sql`, `url`, etc. In our example today, we will make an MCP server that queries a fake customer database. Observations will therefore be of type `sql`.
- `content`: The actual content of the query. For a document, it will be a document chunk. For a SQL query, it will be the returned table as markdown (for instance). It must be a string to be passed to the LLM.
- `metadata`: All data one cannot pass to an LLM, because of its size (you don't want to pay for all those tokens!) or lack of relevance to the question, but key to source the answer. For a document, it will contain the filename, when it was edited and by whom. For a SQL query, it will contain the SQL query actually used by the agent, the actual data returned, the server, the database, etc. The metadata will be what is displayed to the user when citing an observation.


> **Tip:**
> This is similar to a `Document` in LangChain, but more general. Ideally, a future LangChain `Document` would inherit from a generic `Observation`.
{: .notice--info}

## The objective

When an agent makes a call to the MCP server, the server returns the content of the query for the LLM, but also passes lots of metadata to be used for citation. 
Ideally, all this should be done without having to re-engineer an entire MCP server. Here, we will use the `mcp` library as is with `langchain`.

## Let's create a fake database

Let's assume our company needs an agent to query data from a SQL database, and be able to display the observations as sources where trust is paramount.
We are going to create a fake dataset, which a company could have such as a customer database.

We can create a pandas DataFrame, then create a duckdb database for our fake database:

```python 
import pandas as pd
from sqlalchemy import create_engine
import os

db_path = "~/localfiles/databases/data-customer.db"
customer_data = {
    "CustomerID": range(1, 11),
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Helen", "Ian", "Julia"],
    "Age": [25, 34, 28, 45, 32, 41, 29, 38, 27, 36],
    "Email": [
        "alice@example.com", "bob@example.com", "charlie@example.com", "david@example.com",
        "eva@example.com", "frank@example.com", "grace@example.com", "helen@example.com",
        "ian@example.com", "julia@example.com"
    ],
    "Country": ["USA", "Canada", "UK", "Germany", "France", "Italy", "Spain", "Australia", "Japan", "Brazil"]
}
customer_df = pd.DataFrame(customer_data)

if not os.path.exists(db_path):
    eng = create_engine(f"duckdb:///{db_path}")
    with eng.connect() as conn:
        customer_df.to_sql('CUSTOMERS', conn, if_exists='replace', index=False)
```

**Explanation:**
- We create a simple customer DataFrame and store it in a DuckDB database using SQLAlchemy.
- This simulates a real-world customer database for our MCP server to query.

To query the database, one can use sqlalchemy:

```python
from sqlalchemy import text
eng = create_engine(f"duckdb:///{db_path}")
with eng.connect() as conn:
    # Example query
    result = conn.execute(text("SELECT * FROM information_schema.tables")).fetchall()

print(result)
```

This will show us the `CUSTOMERS` table in the database.

## Let's create the MCP server

First, we are going to leverage `langchain-community` with its `SQLDatabase` toolkit. It is very convenient, because the toolkit just uses SQLAlchemy, and exposes SQL-ready tools out of the box, namely:
- `ListSQLDatabaseTool`: Output is the list of table names with their associated descriptions.
- `InfoSQLDatabaseTool`: Get the schema and sample rows for the specified SQL tables.
- `QuerySQLCheckerTool`: Check if query is correct before executing it (leverages a LLM).
- `QuerySQLDatabaseTool`: Execute a SQL query against the database and get back the result.

What we have to do is simply modify those (or make a wrapper) to return an `Observation` instead of just the text data.

> **Tip:**
> You can wrap or extend the LangChain tools to return your custom Observation objects, and LangChain will handle the Pydantic models for you.
{: .notice--info}

I also leverage the `mcp` server as-is. The team made extremely useful [examples](https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/servers/simple-auth/mcp_simple_auth/server.py). I just use those to run a FastMCP server.

To make it easier, I also use `langchain-mcp-adapters` with the `to_fastmcp()` helper. This transforms a LangChain Tool into a FastMCP Tool object automatically.

Thus, most of the hard work is limited to making the LangChain tools return Observations objects. 

**To launch the MCP server (see the repos):**

```bash
source ./load_env.sh # load env variables in the terminal from ~/.env
conda activate sandbox # activate conda env
python server.py # run server with unicorn
```

**Sample output:**
```bash
2025-10-19 19:27:05,899 - src.server.app - INFO - Create app with port 3002, host localhost.
2025-10-19 19:27:05,908 - src.server.app - INFO - Create tools called list_customer_tables, get_customer_table_schema, verify_customer_table_query, query_customer_table_data
2025-10-19 19:27:05,921 - __main__ - INFO - Starting server with streamable-http transport
INFO:     Started server process [92230]
INFO:     Waiting for application startup.
2025-10-19 19:27:05,933 - mcp.server.streamable_http_manager - INFO - StreamableHTTP session manager started
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:3002 (Press CTRL+C to quit)
```

## Test the MCP server with a basic ReACT agent

Before diving into a more advanced agent that leverages Observations returned from the MCP server (Part 2), let's use a simple agent out-of-the-box from LangChain to test the MCP server.

Here, we leverage `langgraph` and `langchain_mcp_adapters`. The ReACT agent is given the url `http://localhost:3002/mcp` with transport `streamable_http` to connect to our MCP server, then all the rest is out-of-the-box in LangGraph.

I have made a simple script:

```bash
source ./load_env.sh # load env variables in the terminal from ~/.env
conda activate sandbox # activate conda env
python client.py --interactive # run agent with pre-built interactive questions
```

**Example output:**

*What is the country of my customer called Alice?*

```bash
What do you want to search? 
>> "What is the country of my customer called Alice?"
================================== Ai Message ==================================
Tool Calls:
  list_customer_tables (call_uz20cW9YDLpUQ4NzcVa36A9M)
 Call ID: call_uz20cW9YDLpUQ4NzcVa36A9M
  Args:
================================= Tool Message =================================
Name: list_customer_tables

{
  "id": null,
  "type": "String",
  "content": "CUSTOMERS -- This table contains customer information, including unique ID, name, age, email, and country. Use this table to retrieve details about customers for analytics, communication, or segmentation purposes.",
  "metadata": {}
}
================================== Ai Message ==================================
Tool Calls:
  get_customer_table_schema (call_88SdFGQbYzLnH1OXNlP1OwKt)
 Call ID: call_88SdFGQbYzLnH1OXNlP1OwKt
  Args:
    table_names: CUSTOMERS
================================= Tool Message =================================
Name: get_customer_table_schema

{
  "id": null,
  "type": "String",
  "content": "CREATE TABLE CUSTOMERS (\n    CustomerID INTEGER,\n    Name VARCHAR,\n    Age INTEGER,\n    Email VARCHAR,\n    Country VARCHAR\n)\n\n/* Add descriptions to CUSTOMERS table and its columns */\nCOMMENT ON TABLE CUSTOMERS IS 'This table contains customer information, including unique ID, name, age, email, and country. Use this table to retrieve details about customers for analytics, communication, or segmentation purposes.';\nCOMMENT ON COLUMN CUSTOMERS.CustomerID IS 'Unique identifier for each customer.';\nCOMMENT ON COLUMN CUSTOMERS.Name IS 'Name of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Age IS 'Age of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Email IS 'Email address of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Country IS 'Country of residence of the customer.';\n\n/*\n3 rows from CUSTOMERS table:\nCustomerID    Name      Age    Email               Country\n1             Alice     25     alice@example.com   USA\n2             Bob       34     bob@example.com     Canada\n3             Charlie   28     charlie@example.com UK\n*/\n",
  "metadata": {
    "query": "CUSTOMERS",
    "result": "CREATE TABLE CUSTOMERS (\n    CustomerID INTEGER,\n    Name VARCHAR,\n    Age INTEGER,\n    Email VARCHAR,\n    Country VARCHAR\n)\n\n/* Add descriptions to CUSTOMERS table and its columns */\nCOMMENT ON TABLE CUSTOMERS IS 'This table contains customer information, including unique ID, name, age, email, and country. Use this table to retrieve details about customers for analytics, communication, or segmentation purposes.';\nCOMMENT ON COLUMN CUSTOMERS.CustomerID IS 'Unique identifier for each customer.';\nCOMMENT ON COLUMN CUSTOMERS.Name IS 'Name of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Age IS 'Age of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Email IS 'Email address of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Country IS 'Country of residence of the customer.';\n\n/*\n3 rows from CUSTOMERS table:\nCustomerID    Name      Age    Email               Country\n1             Alice     25     alice@example.com   USA\n2             Bob       34     bob@example.com     Canada\n3             Charlie   28     charlie@example.com UK\n*/\n"
  }
}
================================== Ai Message ==================================
Tool Calls:
  verify_customer_table_query (call_0Wqf8wsydKJ6Yz6mONHMkCw1)
 Call ID: call_0Wqf8wsydKJ6Yz6mONHMkCw1
  Args:
    query: SELECT Country FROM CUSTOMERS WHERE Name = 'Alice';
================================= Tool Message =================================
Name: verify_customer_table_query

{
  "id": null,
  "type": "Sql",
  "content": "```sql\nSELECT Country FROM CUSTOMERS WHERE Name = 'Alice';\n```",
  "metadata": {
    "query": "SELECT Country FROM CUSTOMERS WHERE Name = 'Alice';",
    "result": "```sql\nSELECT Country FROM CUSTOMERS WHERE Name = 'Alice';\n```"
  }
}
================================== Ai Message ==================================
Tool Calls:
  query_customer_table_data (call_DDw6yw8iK9H6C0bmpWgXP4O2)
 Call ID: call_DDw6yw8iK9H6C0bmpWgXP4O2
  Args:
    query: SELECT Country FROM CUSTOMERS WHERE Name = 'Alice';
================================= Tool Message =================================
Name: query_customer_table_data

{
  "id": null,
  "type": "Sql",
  "content": "| Country   |\n|:----------|\n| USA       |",
  "metadata": {
    "query": "SELECT Country FROM CUSTOMERS WHERE Name = 'Alice';",
    "result": "| Country   |\n|:----------|\n| USA       |"
  }
}
================================== Ai Message ==================================

The country of your customer named Alice is **USA**.
```

The agent retrieved the tools from the MCP server, then called each tool one by one to get to the answer. For each tool, a dictionary is returned (the Observation), that the LLM uses to provide an answer. A more advanced agent could just pass the "content" to the LLM alongside an "id", and use this to ground an answer (see Part 2).

*More complicated question:*

```bash
What do you want to search? 
>> "List all my customers ID, name and age who are aged above 40. Those are key customers."
================================== Ai Message ==================================
Tool Calls:
  list_customer_tables (call_V7pzFTuVPOlPL5IwSsTrAt2M)
 Call ID: call_V7pzFTuVPOlPL5IwSsTrAt2M
  Args:
================================= Tool Message =================================
Name: list_customer_tables

{
  "id": null,
  "type": "String",
  "content": "CUSTOMERS -- This table contains customer information, including unique ID, name, age, email, and country. Use this table to retrieve details about customers for analytics, communication, or segmentation purposes.",
  "metadata": {}
}
================================== Ai Message ==================================
Tool Calls:
  get_customer_table_schema (call_CaUYWh9YTIQt5LAWXzuDow4Z)
 Call ID: call_CaUYWh9YTIQt5LAWXzuDow4Z
  Args:
    table_names: CUSTOMERS
================================= Tool Message =================================
Name: get_customer_table_schema

{
  "id": null,
  "type": "String",
  "content": "CREATE TABLE CUSTOMERS (\n    CustomerID INTEGER,\n    Name VARCHAR,\n    Age INTEGER,\n    Email VARCHAR,\n    Country VARCHAR\n)\n\n/* Add descriptions to CUSTOMERS table and its columns */\nCOMMENT ON TABLE CUSTOMERS IS 'This table contains customer information, including unique ID, name, age, email, and country. Use this table to retrieve details about customers for analytics, communication, or segmentation purposes.';\nCOMMENT ON COLUMN CUSTOMERS.CustomerID IS 'Unique identifier for each customer.';\nCOMMENT ON COLUMN CUSTOMERS.Name IS 'Name of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Age IS 'Age of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Email IS 'Email address of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Country IS 'Country of residence of the customer.';\n\n/*\n3 rows from CUSTOMERS table:\nCustomerID    Name      Age    Email               Country\n1             Alice     25     alice@example.com   USA\n2             Bob       34     bob@example.com     Canada\n3             Charlie   28     charlie@example.com UK\n*/\n",
  "metadata": {
    "query": "CUSTOMERS",
    "result": "CREATE TABLE CUSTOMERS (\n    CustomerID INTEGER,\n    Name VARCHAR,\n    Age INTEGER,\n    Email VARCHAR,\n    Country VARCHAR\n)\n\n/* Add descriptions to CUSTOMERS table and its columns */\nCOMMENT ON TABLE CUSTOMERS IS 'This table contains customer information, including unique ID, name, age, email, and country. Use this table to retrieve details about customers for analytics, communication, or segmentation purposes.';\nCOMMENT ON COLUMN CUSTOMERS.CustomerID IS 'Unique identifier for each customer.';\nCOMMENT ON COLUMN CUSTOMERS.Name IS 'Name of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Age IS 'Age of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Email IS 'Email address of the customer.';\nCOMMENT ON COLUMN CUSTOMERS.Country IS 'Country of residence of the customer.';\n\n/*\n3 rows from CUSTOMERS table:\nCustomerID    Name      Age    Email               Country\n1             Alice     25     alice@example.com   USA\n2             Bob       34     bob@example.com     Canada\n3             Charlie   28     charlie@example.com UK\n*/\n"
  }
}
================================== Ai Message ==================================
Tool Calls:
  verify_customer_table_query (call_dSxaNtf13u2joR9E3sGzCCfN)
 Call ID: call_dSxaNtf13u2joR9E3sGzCCfN
  Args:
    query: SELECT CustomerID, Name, Age FROM CUSTOMERS WHERE Age > 40;
================================= Tool Message =================================
Name: verify_customer_table_query

{
  "id": null,
  "type": "Sql",
  "content": "```sql\nSELECT CustomerID, Name, Age FROM CUSTOMERS WHERE Age > 40;\n```",
  "metadata": {
    "query": "SELECT CustomerID, Name, Age FROM CUSTOMERS WHERE Age > 40;",
    "result": "```sql\nSELECT CustomerID, Name, Age FROM CUSTOMERS WHERE Age > 40;\n```"
  }
}
================================== Ai Message ==================================
Tool Calls:
  query_customer_table_data (call_5w6R6uYUpxqO2ESICL09OE0f)
 Call ID: call_5w6R6uYUpxqO2ESICL09OE0f
  Args:
    query: SELECT CustomerID, Name, Age FROM CUSTOMERS WHERE Age > 40;
================================= Tool Message =================================
Name: query_customer_table_data

{
  "id": null,
  "type": "Sql",
  "content": "|   CustomerID | Name   |   Age |\n|-------------:|:-------|------:|\n|            4 | David  |    45 |\n|            6 | Frank  |    41 |",
  "metadata": {
    "query": "SELECT CustomerID, Name, Age FROM CUSTOMERS WHERE Age > 40;",
    "result": "|   CustomerID | Name   |   Age |\n|-------------:|:-------|------:|\n|            4 | David  |    45 |\n|            6 | Frank  |    41 |"
  }
}
================================== Ai Message ==================================

Here are the details of your key customers aged above 40:

| CustomerID | Name  | Age |
|------------|-------|-----|
| 4          | David | 45  |
| 6          | Frank | 41  |
```

Here, the agent was capable of building a valid SQL query by providing the schema of the table. The key is to provide a clear definition at the tool level (to pick the right one), then at the table level (to pick the right one), then at the columns/schema levels. When an agent fails, it is because it picks the wrong tool or the wrong table. Thus, each tool and table must be easy to discriminate. If you have two customer databases to search, then the agent will pick the wrong one from time to time. You must have one tool and one unified customer table in your data warehouse. This for everything so it is obvious which tool and which table to use for a question.

## Summary & Next Steps

In this post, we built a simple MCP server that returns observations for every answer, supporting scientific rigor and reproducibility. We also showed how to connect a basic ReACT agent to this server. In Part 2, we'll build a more advanced agent that leverages these observations for grounded trustworthy answers.

*Questions or suggestions? Leave a comment below or reach out on GitHub!*
