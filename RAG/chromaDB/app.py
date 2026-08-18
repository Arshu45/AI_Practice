# ChromaDB client initialization


import chromadb
chroma_client = chromadb.Client()

collection_name = "my_collection"

# Create Collection if it doesn't exist
collection = chroma_client.get_or_create_collection(name=collection_name)

documents = [
    {
        "id": "doc1",
        "content": "This is the content of document 1. My name is Arsh",
        "metadata": {"source": "source1"}
    },
    {
        "id": "doc2",
        "content": "This is the content of document 2.",
        "metadata": {"source": "source2"}
    },
    {
        "id": "doc3",
        "content": "This is the content of document 3.",
        "metadata": {"source": "source3"}
    }
]

# Add documents to the collection
for doc in documents:
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["content"]],
        metadatas=[doc["metadata"]]
    )
# Define a Query Text
query_text = "My name is arsh"


# Perform a similarity search
results = collection.query(
    query_texts=[query_text],
    n_results=2,
    include=["documents", "metadatas"]
)

print(results)