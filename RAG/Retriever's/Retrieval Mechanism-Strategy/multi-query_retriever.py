# Imports

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
import os


# Relevant documents

all_docs = [

    # ---------------- HEALTH & ENERGY ----------------

    Document(
        page_content="Regular physical activity such as walking, cycling, and swimming can improve cardiovascular fitness, increase stamina, and support consistent energy throughout the day.",
        metadata={"source": "H1", "category": "health", "topic": "physical_activity"}
    ),

    Document(
        page_content="Drinking enough water is important for maintaining normal body functions. Even mild dehydration can contribute to tiredness, headaches, and difficulty concentrating.",
        metadata={"source": "H2", "category": "health", "topic": "hydration"}
    ),

    Document(
        page_content="Eating balanced meals containing whole grains, vegetables, fruits, lean proteins, and healthy fats provides nutrients that support sustained energy rather than short bursts followed by fatigue.",
        metadata={"source": "H3", "category": "health", "topic": "nutrition"}
    ),

    Document(
        page_content="Consistent sleep is essential for physical recovery, memory, concentration, and emotional regulation. Most adults benefit from maintaining a regular sleep schedule and getting adequate sleep each night.",
        metadata={"source": "H4", "category": "health", "topic": "sleep"}
    ),

    Document(
        page_content="Mindfulness, meditation, and controlled breathing exercises can help people manage stress and improve attention. These practices are often used as complementary techniques for maintaining mental wellbeing.",
        metadata={"source": "H5", "category": "health", "topic": "mental_wellness"}
    ),

    Document(
        page_content="Excessive consumption of highly processed foods and added sugars can produce rapid changes in blood glucose and may make some people feel tired after an initial increase in energy.",
        metadata={"source": "H6", "category": "health", "topic": "nutrition"}
    ),

    Document(
        page_content="Taking short movement breaks during long periods of sitting can reduce physical stiffness and help maintain alertness. Brief walks or stretching can be incorporated into a normal workday.",
        metadata={"source": "H7", "category": "health", "topic": "physical_activity"}
    ),

    Document(
        page_content="Caffeine can temporarily increase alertness and reduce the perception of fatigue, but consuming large amounts or using caffeine late in the day can interfere with sleep and indirectly affect energy levels.",
        metadata={"source": "H8", "category": "health", "topic": "energy_management"}
    ),

    Document(
        page_content="A sustainable approach to wellbeing combines physical activity, adequate sleep, nutritious food, hydration, stress management, and realistic daily routines instead of relying on a single intervention.",
        metadata={"source": "H9", "category": "health", "topic": "overall_wellness"}
    ),

    Document(
        page_content="Maintaining a consistent daily routine can make it easier to balance work, exercise, meals, sleep, and relaxation. Predictable routines may also reduce decision fatigue.",
        metadata={"source": "H10", "category": "health", "topic": "lifestyle"}
    ),


    # ---------------- MENTAL HEALTH & STRESS ----------------

    Document(
        page_content="Chronic stress can affect sleep quality, concentration, mood, and perceived energy. Managing stress through healthy routines can therefore contribute to better overall functioning.",
        metadata={"source": "M1", "category": "mental_health", "topic": "stress"}
    ),

    Document(
        page_content="Deep breathing exercises can be used as a simple relaxation technique. Slowing the breathing pattern and focusing attention on the breath may help people manage moments of tension.",
        metadata={"source": "M2", "category": "mental_health", "topic": "breathing"}
    ),

    Document(
        page_content="Social connection and supportive relationships can contribute to emotional wellbeing. Spending meaningful time with friends, family, or community groups can help reduce feelings of isolation.",
        metadata={"source": "M3", "category": "mental_health", "topic": "social_wellbeing"}
    ),

    Document(
        page_content="Taking regular breaks from demanding cognitive tasks can help reduce mental fatigue. Breaks are particularly useful when work requires sustained concentration for long periods.",
        metadata={"source": "M4", "category": "mental_health", "topic": "mental_fatigue"}
    ),

    Document(
        page_content="Setting realistic goals and dividing large tasks into smaller steps can reduce feelings of being overwhelmed. Clear priorities can also make it easier to maintain a balanced daily schedule.",
        metadata={"source": "M5", "category": "mental_health", "topic": "productivity"}
    ),


    # ---------------- WORK & PRODUCTIVITY ----------------

    Document(
        page_content="Prioritizing important tasks at the beginning of the workday can help people focus their attention on high-value activities before dealing with less important tasks.",
        metadata={"source": "P1", "category": "productivity", "topic": "prioritization"}
    ),

    Document(
        page_content="Time-blocking divides the workday into dedicated periods for specific activities. This approach can reduce context switching and help people protect time for focused work.",
        metadata={"source": "P2", "category": "productivity", "topic": "time_management"}
    ),

    Document(
        page_content="Multitasking can increase the mental cost of switching between unrelated tasks. Working on one important activity at a time can make sustained concentration easier.",
        metadata={"source": "P3", "category": "productivity", "topic": "focus"}
    ),

    Document(
        page_content="Taking short breaks during intensive computer work can help maintain concentration and reduce mental fatigue. A balanced schedule alternates periods of focused work with recovery.",
        metadata={"source": "P4", "category": "productivity", "topic": "breaks"}
    ),

    Document(
        page_content="Good productivity is not simply about working longer hours. Rest, sleep, prioritization, and sustainable workloads are important for maintaining performance over longer periods.",
        metadata={"source": "P5", "category": "productivity", "topic": "sustainable_productivity"}
    ),


    # ---------------- TECHNOLOGY ----------------

    Document(
        page_content="Python is a general-purpose programming language known for readable syntax and a large ecosystem of libraries for web development, data analysis, automation, and machine learning.",
        metadata={"source": "T1", "category": "technology", "topic": "python"}
    ),

    Document(
        page_content="Vector databases store numerical representations of information and allow applications to search for documents based on semantic similarity rather than relying only on exact keyword matches.",
        metadata={"source": "T2", "category": "technology", "topic": "vector_database"}
    ),

    Document(
        page_content="Embeddings convert text, images, or other information into numerical vectors. Similar concepts can have vectors that are closer together in the embedding space.",
        metadata={"source": "T3", "category": "technology", "topic": "embeddings"}
    ),

    Document(
        page_content="FAISS is a library designed for efficient similarity search over dense vectors. It is commonly used in applications that need to retrieve information based on embeddings.",
        metadata={"source": "T4", "category": "technology", "topic": "faiss"}
    ),

    Document(
        page_content="Retrieval-Augmented Generation combines information retrieval with language generation. A retriever first finds relevant documents, and a language model can then use those documents as context.",
        metadata={"source": "T5", "category": "technology", "topic": "rag"}
    ),


    # ---------------- SCIENCE ----------------

    Document(
        page_content="Photosynthesis allows plants to convert light energy into chemical energy. Plants use sunlight, carbon dioxide, and water to produce carbohydrates while releasing oxygen.",
        metadata={"source": "S1", "category": "science", "topic": "photosynthesis"}
    ),

    Document(
        page_content="Black holes are regions of spacetime where gravity is extremely strong. The boundary around a black hole beyond which light cannot escape is called the event horizon.",
        metadata={"source": "S2", "category": "science", "topic": "astronomy"}
    ),

    Document(
        page_content="The human body obtains energy from nutrients such as carbohydrates, fats, and proteins. These nutrients are metabolized through complex biochemical processes that support cellular functions.",
        metadata={"source": "S3", "category": "science", "topic": "human_metabolism"}
    ),


    # ---------------- BUSINESS & ECONOMICS ----------------

    Document(
        page_content="A company's revenue represents the income generated from selling its products or services. Profit is calculated after subtracting relevant costs and expenses from revenue.",
        metadata={"source": "B1", "category": "business", "topic": "finance"}
    ),

    Document(
        page_content="Customer retention measures how effectively a business keeps its existing customers over time. Improving retention can reduce the need to constantly acquire new customers.",
        metadata={"source": "B2", "category": "business", "topic": "customer_retention"}
    ),

    Document(
        page_content="Market research helps organizations understand customer needs, competitor behavior, market trends, and potential opportunities before making business decisions.",
        metadata={"source": "B3", "category": "business", "topic": "market_research"}
    ),
]


# Load Environment Variables
load_dotenv()



# Initialize Local Hugging Face LLM
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

generator = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    max_new_tokens=128,
    temperature=0.2
)

chat_model = HuggingFacePipeline(
    pipeline=generator
)



# Initialize Hugging Face Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create FAISS Vector Store
vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model
)


# Create Similarity Retriever
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# Create Multi-Query Retriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 5}
    ),
    llm=chat_model
)


# Define Query
query = "How can I improve my energy levels and maintain a healthy balance in daily life?"
# query1 = "How can I improve my energy levels?"
# query2 = "What lifestyle habits can improve my physical and mental wellbeing?"
# query3 = "How can I maintain balance between work, rest, exercise, and mental health?"

# Retrieve results
similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n--- Similarity Result {i + 1} ---")
    print(doc.page_content)



print("*" * 150)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Multi-Query Result {i + 1} ---")
    print(doc.page_content)