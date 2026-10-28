from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query(
    "What is machine learning?"
)
txt=['what is ML',
    'superwised algo is a ML algo']

vector_v2= embeddings.embed_documents(txt)

# print(vector)
print("Vector dimensions:", len(vector))
print("vector_v2 dimensions: ",len(vector_v2))