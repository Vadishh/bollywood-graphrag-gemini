\# Bollywood GraphRAG with Gemini



A Graph Retrieval-Augmented Generation (GraphRAG) application for querying a Bollywood knowledge graph in natural language.



The application combines Neo4j graph traversal, Gemini embeddings, and Gemini answer generation to provide grounded responses about movies, actors, directors, production houses, composers, and awards.



\## Features



\- Neo4j knowledge graph with Bollywood entities and relationships

\- Gemini embeddings for semantic search

\- Exact entity matching for more reliable retrieval

\- Multi-hop graph traversal for contextual answers

\- FastAPI backend with REST endpoints

\- Streamlit chat interface

\- Docker Compose support



\## Tech Stack



\- Python

\- Neo4j

\- Google Gemini API

\- FastAPI

\- Streamlit

\- Docker



\## Architecture



```text

User question

&#x20;   ↓

Exact entity matching + Gemini semantic search

&#x20;   ↓

Neo4j graph traversal

&#x20;   ↓

Relevant graph context

&#x20;   ↓

Gemini answer generation

&#x20;   ↓

Grounded response

```



\## Run Locally



\### 1. Configure environment variables



```powershell

Copy-Item .env.example .env

```



Add your Gemini API key to `.env`:



```env

GEMINI\_API\_KEY=your-gemini-key-here

```



\### 2. Install and load the graph



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

docker compose up neo4j -d

cd src

python loader.py

python embeddings.py

```



\### 3. Start the application



In one terminal:



```powershell

cd src

uvicorn api:app --reload --port 8000

```



In another terminal:



```powershell

cd src

streamlit run app.py

```



Open `http://localhost:8501`.



\## Example Questions



\- Which movies were directed by Rajkumar Hirani?

\- Which movies were produced by Dharma Productions?

\- Which films did Aamir Khan act in?

\- Which movies produced by Yash Raj Films have Shah Rukh Khan?

\- What awards did Dangal win?

