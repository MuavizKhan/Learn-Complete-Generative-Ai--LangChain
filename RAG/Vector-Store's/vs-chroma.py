from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path


# 1. Create Documents

doc1 = Document(
    page_content=(
        "Virat Kohli is one of the most successful and consistent "
        "batsmen in IPL history. Known for his aggressive batting style "
        "and fitness, he has led the Royal Challengers Bangalore in "
        "multiple seasons."
    ),
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content=(
        "Rohit Sharma is the most successful captain in IPL history, "
        "leading Mumbai Indians to five titles. He's known for his calm "
        "demeanor and ability to play big innings under pressure."
    ),
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content=(
        "MS Dhoni, famously known as Captain Cool, has led Chennai Super "
        "Kings to multiple IPL titles. His finishing skills, wicketkeeping, "
        "and leadership are legendary."
    ),
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content=(
        "Jasprit Bumrah is considered one of the best fast bowlers in "
        "T20 cricket. Playing for Mumbai Indians, he is known for his "
        "yorkers and death-over expertise."
    ),
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content=(
        "Ravindra Jadeja is a dynamic all-rounder who contributes with "
        "both bat and ball. Representing Chennai Super Kings, his quick "
        "fielding and match-winning performances make him a key player."
    ),
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]


# 2. Document IDs

ids = [
    "virat_kohli",
    "rohit_sharma",
    "ms_dhoni",
    "jasprit_bumrah",
    "ravindra_jadeja"
]


# 3. Chroma Database Location

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "my_chroma_db"


# 4. Embedding Model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 5. Create Vector Store

vector_store = Chroma(
    collection_name="sample",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)


# 6. Add Documents

vector_store.add_documents(
    documents=docs,
    ids=ids
)


# 7. View Documents

result = vector_store.get(
    include=["documents", "metadatas"]
)

print("\nStored Documents:")
print(result)


# 8. Similarity Search

results = vector_store.similarity_search(
    query="Who among these are bowlers?",
    k=2
)

print("\nSimilarity Search:")

for doc in results:
    print(doc.page_content)
    print(doc.metadata)


# 9. Similarity Search With Score

results_with_score = vector_store.similarity_search_with_score(
    query="Who among these are bowlers?",
    k=2
)

print("\nSimilarity Search With Score:")

for doc, score in results_with_score:
    print("Score:", score)
    print("Document:", doc.page_content)
    print("Metadata:", doc.metadata)
    print()


# 10. Metadata Filtering

results = vector_store.similarity_search(
    query="",
    k=5,
    filter={"team": "Chennai Super Kings"}
)

print("\nChennai Super Kings Players:")

for doc in results:
    print(doc.page_content)
    print(doc.metadata)


# 11. Update Document

updated_doc1 = Document(
    page_content=(
        "Virat Kohli, the former captain of Royal Challengers Bangalore "
        "(RCB), is renowned for his aggressive leadership and consistent "
        "batting performances. He holds the record for the most runs in "
        "IPL history, including multiple centuries in a single season."
    ),
    metadata={
        "team": "Royal Challengers Bangalore"
    }
)

vector_store.update_document(
    document_id="virat_kohli",
    document=updated_doc1
)


# 12. Delete Document

# Uncomment this when you want to delete Virat Kohli

# vector_store.delete(
#     ids=["virat_kohli"]
# )


# 13. Final View

print("\nFinal Vector Store:")

print(
    vector_store.get(
        include=["documents", "metadatas"]
    )
)