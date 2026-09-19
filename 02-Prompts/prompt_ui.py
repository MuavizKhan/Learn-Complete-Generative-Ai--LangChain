from dotenv import load_dotenv
from pathlib import Path

import streamlit as st

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt


# Load environment variables
load_dotenv()


# Hugging Face hosted model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)


# Streamlit UI
st.header("Research Tool")


paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)


style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical",
    ],
)


length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ],
)


# Load prompt template
template = load_prompt(
    str(Path(__file__).with_name("template.json"))
)


# Generate explanation - use 'chain'
if st.button("Summarize"):

    chain = template | model

    try:

        result = chain.invoke(
            {
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input,
            }
        )

        st.write(result.content)

    except Exception as error:

        st.error(
            f"Unable to generate the explanation: {error}"
        )