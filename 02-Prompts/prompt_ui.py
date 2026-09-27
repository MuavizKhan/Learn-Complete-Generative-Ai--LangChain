# ---------------------------------------------------------------------------
# prompt_ui.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# The last piece of 02-Prompts, pulling several ideas together into an
# actual clickable app instead of a terminal script:
#   - Streamlit turns plain Python variables into a browser UI -- no HTML.
#   - load_prompt() loads a PromptTemplate that was saved to disk as JSON
#     (template.json) instead of being written inline in Python -- the
#     "single message, Dynamic" cell from the grid a few files back,
#     finally shown in its own file.
#   - `template | model` is LCEL: piping a template straight into a model
#     so ONE .invoke() does both steps at once. Every earlier file called
#     .invoke() on a template, then separately called .invoke() on a
#     model. This is your first real look at chaining -- the entire
#     subject of 03-Chains and 04-Runnables, arriving a little early.
#
# Run this with `streamlit run prompt_ui.py`, not `python prompt_ui.py` --
# a plain Python run won't start the web server Streamlit needs to show
# anything.
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
from pathlib import Path

import streamlit as st

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt


# Load environment variables
load_dotenv()


# WHY @st.cache_resource?
# Streamlit reruns this ENTIRE script top to bottom on every interaction --
# picking a different paper, a different style, anything, not just
# clicking Summarize. Without this decorator, the model client gets
# rebuilt from scratch every single time, for no reason. This tells
# Streamlit to build it once and reuse the same object on every rerun.
@st.cache_resource
def get_model():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.3,
    )
    return ChatHuggingFace(llm=llm)


model = get_model()


# Streamlit UI
st.header("Research Tool")


# "Select..." as the first option means the dropdown doesn't quietly land
# on a real paper before anyone has actually chosen one -- your own notes'
# version of this exact snippet has this option; the file in the repo had
# dropped it, which means loading the page already has a paper "selected"
# by accident.
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
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

    if paper_input == "Select...":
        st.warning("Please choose a research paper first.")
        st.stop()

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