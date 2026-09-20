from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.5
)

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.

Artificial intelligence is transforming modern healthcare. Machine learning models can analyze medical images, assist doctors in diagnosis, and identify patterns in large healthcare datasets.

Renewable energy is another important area of technological development. Solar panels, wind turbines, and battery storage systems are helping reduce dependence on traditional fossil fuels.
"""

docs = text_splitter.create_documents([text])
print(len(docs))
print(docs)