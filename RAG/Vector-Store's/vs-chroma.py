# ---------------------------------------------------------------------------
# vs-chroma.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# Everything so far (01-Models/EmbeddingModels) turned text into vectors and
# stopped there -- you got numbers back and the program ended. A Vector
# Store is what you do with those vectors AFTERWARDS: it's a database built
# specifically to hold (text + its vector + metadata) together, and to
# answer "which of these stored pieces of text is closest in meaning to
# THIS new query" quickly, even across millions of entries.
#
# Chroma is one specific vector store -- free, open-source, and it can save
# its data to a folder on your own disk (no server, no account, no API key).
#
# This file walks through the full lifecycle: create documents, store them,
# read them back, search them by meaning, search them with a metadata
# filter, update one, and (optionally) delete one.
# ---------------------------------------------------------------------------

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Create Documents
#
# WHY a Document, instead of the plain strings we used in EmbeddingModels?
# A Document bundles two things together: page_content (the actual text)
# and metadata (structured facts ABOUT that text). We're storing each
# player's team as metadata here specifically so we can filter by it later,
# in step 10 -- that's not possible with a plain string.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Document IDs
#
# WHY explicit IDs? Without them, Chroma auto-generates a random ID per
# document, which you'd have no way to refer back to. Giving each document
# a readable ID of our own choosing is what makes step 11 (update this ONE
# specific document) and step 12 (delete this ONE specific document)
# possible at all.
# ---------------------------------------------------------------------------

ids = [
    "virat_kohli",
    "rohit_sharma",
    "ms_dhoni",
    "jasprit_bumrah",
    "ravindra_jadeja"
]


# ---------------------------------------------------------------------------
# 3. Chroma Database Location
#
# WHY Path(__file__).resolve().parent instead of a plain string path?
# It always points next to THIS script, no matter which folder you run it
# from -- so the database ends up in the same place for you as it would
# for anyone else who clones this repo, on any operating system.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "my_chroma_db"


# ---------------------------------------------------------------------------
# 4. Embedding Model
#
# The exact same class and model as 01-Models/EmbeddingModels -- Chroma
# doesn't do its own embedding. It calls this model to convert text to
# vectors, and just handles the storing/searching of the result.
# ---------------------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------------------------
# 5. Create Vector Store
#
# collection_name groups related documents together inside one Chroma
# database (like a table inside a database). persist_directory is what
# makes this survive between runs -- rerun this file tomorrow and your
# documents are still there, instead of starting from zero every time.
# ---------------------------------------------------------------------------

try:
    vector_store = Chroma(
        collection_name="sample",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # -----------------------------------------------------------------
    # 6. Add Documents
    #
    # This is the step that actually calls the embedding model -- each
    # document's page_content gets converted to a vector here, and the
    # (text, vector, metadata, id) bundle gets written to disk.
    # -----------------------------------------------------------------

    vector_store.add_documents(
        documents=docs,
        ids=ids
    )

    # -----------------------------------------------------------------
    # 7. View Documents
    #
    # .get() is a plain lookup, not a search -- it just dumps back
    # everything currently stored, so you can confirm step 6 worked.
    # -----------------------------------------------------------------

    result = vector_store.get(
        include=["documents", "metadatas"]
    )

    print("\nStored Documents:")
    print(result)

    # -----------------------------------------------------------------
    # 8. Similarity Search
    #
    # THE actual point of a vector store: none of these documents contain
    # the word "bowlers", yet the model finds Jasprit Bumrah anyway,
    # because "fast bowler" and "yorkers" are close in MEANING to
    # "bowlers" even without matching the word itself. A keyword search
    # would have found nothing here -- this is exactly the gap embeddings
    # exist to close.
    # -----------------------------------------------------------------

    results = vector_store.similarity_search(
        query="Who among these are bowlers?",
        k=2
    )

    print("\nSimilarity Search:")

    for doc in results:
        print(doc.page_content)
        print(doc.metadata)

    # -----------------------------------------------------------------
    # 9. Similarity Search With Score
    #
    # Same search, but this version also hands back HOW close each
    # match was. Lower distance = more similar. Useful once you want to
    # decide "is this actually a good match, or just the least-bad one?"
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 10. Metadata Filtering
    #
    # query="" is deliberate, not a placeholder left in by mistake: this
    # step isn't asking "which is closest in meaning" at all -- filter
    # does the entire job by itself, restricting results to Chennai
    # Super Kings documents only. The empty query just satisfies the
    # method's required argument without adding any semantic search on
    # top of the filter.
    # -----------------------------------------------------------------

    results = vector_store.similarity_search(
        query="",
        k=5,
        filter={"team": "Chennai Super Kings"}
    )

    print("\nChennai Super Kings Players:")

    for doc in results:
        print(doc.page_content)
        print(doc.metadata)

    # -----------------------------------------------------------------
    # 11. Update Document
    #
    # Documents aren't frozen once stored. update_document re-embeds the
    # new text and replaces the old vector under the same ID -- Virat
    # Kohli's entry changes in place instead of becoming a duplicate.
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 12. Delete Document
    #
    # Uncomment this when you want to delete Virat Kohli
    # -----------------------------------------------------------------

    # vector_store.delete(
    #     ids=["virat_kohli"]
    # )

    # -----------------------------------------------------------------
    # 13. Final View
    # -----------------------------------------------------------------

    print("\nFinal Vector Store:")

    print(
        vector_store.get(
            include=["documents", "metadatas"]
        )
    )

except Exception as e:
    print(f"Something went wrong working with the vector store: {e}")