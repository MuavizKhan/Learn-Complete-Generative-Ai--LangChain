# ---------------------------------------------------------------------------
# stroutputparser.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# StrOutputParser is the simplest output parser in LangChain: it takes
# whatever a Chat Model hands back (an AIMessage object) and pulls out just
# the .content string. You've been doing this by hand in every ChatModels
# file so far -- result.content, every time. A parser does that same
# extraction automatically, as one more link in a chain, so you never have
# to remember to write .content yourself again.
#
# That's also the real point of this file: template | model | parser is
# prompt_ui.py's two-stage chain (template | model) with one more link
# added. Three Runnables, one .invoke() call. And this file goes one step
# further than prompt_ui.py did -- it builds TWO such chains, then feeds
# the first chain's plain-string output straight into the second chain's
# input. Running one chain's result through a second chain is exactly what
# the next module, 03-Chains, is built around -- this file is a genuine
# preview of it, sitting inside Output Parsers.
# ---------------------------------------------------------------------------

from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# Same model and free-tier note as messages.py / chatbot.py.
# max_new_tokens is worth setting explicitly here specifically: the first
# prompt below asks for a "detailed report," and without an explicit,
# generous cap, a long generation can get cut off mid-sentence on
# whatever the default happens to be -- which would then feed a silently
# truncated report into the summarization step that follows it.
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)


# First prompt -> Generate detailed report
template1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a detailed report on {topic}"
)


# Second prompt -> Summarize the report
template2 = PromptTemplate(
    input_variables=["text"],
    template="Write a summary of the following text:\n{text}"
)


parser = StrOutputParser()

# using CHAIN's
chain1 = template1 | model | parser
chain2 = template2 | model | parser


try:
    result = chain1.invoke({
        "topic": "Black Hole"
    })

    # result here is already a plain string, not an AIMessage -- parser
    # already did that extraction inside chain1. That's what makes handing
    # it straight to chain2 as {"text": result} work correctly.
    result1 = chain2.invoke({
        "text": result
    })

    print("\n========== FINAL SUMMARY ==========\n")
    print(result1)

except Exception as e:
    print(f"Something went wrong running the chain: {e}")