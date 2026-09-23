import wikipediaapi
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class WikipediaRetriever(BaseRetriever):

    wiki: object
    top_k_results: int = 2

    def _get_relevant_documents(self, query: str, *, run_manager=None):

        titles = [
            "History of India",
            "History of Pakistan",
            "India-Pakistan relations",
            "China-India relations",
            "Sino-Pakistani relations"
        ]

        docs = []

        for title in titles[:self.top_k_results]:

            page = self.wiki.page(title)

            if page.exists():

                docs.append(
                    Document(
                        page_content=page.text,
                        metadata={
                            "title": page.title,
                            "source": "Wikipedia",
                            "url": page.fullurl
                        }
                    )
                )

        return docs


wiki = wikipediaapi.Wikipedia(
    user_agent="LangChainRAGDemo/1.0 (your_email@example.com)",
    language="en"
)

retriever = WikipediaRetriever(
    wiki=wiki,
    top_k_results=2
)

query = "geopolitical history of India and Pakistan from the perspective of China"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):

    print(f"\n--- Result {i + 1} ---")
    print(f"Title: {doc.metadata['title']}")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content:\n{doc.page_content[:2000]}...")