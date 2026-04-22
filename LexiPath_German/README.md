# LexiPath German

State-aware German tutoring project for HCNLP.

## Setup

Download project zip or clone project repository
Create virtual environment with the command navigating to proper folder : 
```bash
python -m venv venv
venv\Scripts\activate
source venv/bin/activate
pip install -r requirements.txt
Run this command : pip install -m requirements.txt
Run this command : pip install langchain-postgres langchain-huggingface psycopg[binary] langchain-openai
Run this command: pip install langchain-huggingface
Run this command: python adder.py
python retriever.py
python app.py
```

## PGVector note

The project expects a PostgreSQL server with `pgvector` enabled. If you get erro such as no PostgreSQL server running on localhost:5432, then start a container that matches those credentials:

docker run --name lexipath-pgvector \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  -d pgvector/pgvector:pg17

Create a Moonshot/Kimi API key in the Kimi platform:

Go to: https://platform.moonshot.ai
Sign up / log in
Navigate to API Keys
Create a key → copy it


Set MOONSHOT_API_KEY:

Mac / Linux (Terminal)-
export MOONSHOT_API_KEY="your_api_key_here"
Windows (PowerShell)-
setx MOONSHOT_API_KEY "your_api_key_here"



