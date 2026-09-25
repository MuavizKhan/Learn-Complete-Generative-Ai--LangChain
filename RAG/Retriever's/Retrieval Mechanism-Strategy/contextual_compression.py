from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from transformers import pipeline


# Documents
docs = [
    Document(
        page_content=(
            "The Grand Canyon is one of the most visited natural wonders in the world. "
            "Photosynthesis is the process by which green plants convert sunlight into energy. "
            "Millions of tourists travel to see it every year. "
            "The rocks date back millions of years."
        ),
        metadata={"source": "Doc1"}
    ),

    Document(
        page_content=(
            "In medieval Europe, castles were built primarily for defense. "
            "The chlorophyll in plant cells captures sunlight during photosynthesis. "
            "Knights wore armor made of metal. "
            "Siege weapons were often used to breach castle walls."
        ),
        metadata={"source": "Doc2"}
    ),

    Document(
        page_content=(
            "Basketball was invented by Dr. James Naismith in the late 19th century. "
            "It was originally played with a soccer ball and peach baskets. "
            "NBA is now a global league."
        ),
        metadata={"source": "Doc3"}
    ),

    Document(
        page_content=(
            "The history of cinema began in the late 1800s. "
            "Silent films were the earliest form. "
            "Thomas Edison was among the pioneers. "
            "Photosynthesis does not occur in animal cells. "
            "Modern filmmaking involves complex CGI and sound design."
        ),
        metadata={"source": "Doc4"}
    )
]


# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Vector Store
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)


# Base Retriever
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# Local Hugging Face LLM
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=64,
    do_sample=False,
    return_full_text=False
)

llm = HuggingFacePipeline(
    pipeline=generator
)


# Contextual Compressor
compressor = LLMChainExtractor.from_llm(llm)


# Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)


# Query
query = "What is photosynthesis?"

compressed_results = compression_retriever.invoke(query)


# Results
for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content)