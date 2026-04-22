from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

CONNECTION_STRING = "postgresql+psycopg://postgres:mypassword@localhost:5432/postgres"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="lexipath_grammar", # Must match your adder.py
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

# This deletes all documents in the specific collection
vector_store.delete_collection()
print("Collection 'lexipath_grammar' has been cleared.")