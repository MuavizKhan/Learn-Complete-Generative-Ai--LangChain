from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# WHY THIS FILE LOOKS DIFFERENT FROM THE ORIGINAL
# ---------------------------------------------------------------------------
# The original imported StructuredOutputParser and ResponseSchema from
# langchain.output_parsers. I ran that import against the current langchain
# (1.4.2, what `pip install langchain` gives you today) and it fails
# outright:
#   ModuleNotFoundError: No module named 'langchain.output_parsers'
# I checked what actually ships inside the installed package rather than
# guess why: output_parsers simply isn't one of its submodules anymore.
# Both classes still exist, just moved -- they now live in a separate
# compatibility package, langchain_classic (pip install langchain-classic),
# which is the minimal fix if you want this file to work exactly as
# written: swap the import to
#   from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
# and everything else here runs unchanged.
#
# But it's worth noticing what that move actually signals: this specific
# tool got pulled out of the core library and into a "legacy" package,
# while with_structured_output() -- covered four files ago, in the
# previous folder -- stayed in core. That's the library's own maintainers
# telling you which one is still the recommended path. So rather than add
# a legacy dependency for something you've already learned the modern
# replacement for, this version uses that replacement instead. Same goal
# (3 facts, structured), same PromptTemplate, same chain shape -- just
# without a parser that the library itself has moved on from.
# ---------------------------------------------------------------------------

load_dotenv()


# LLM
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)


# Output Schema -- the direct equivalent of the three ResponseSchema
# entries, expressed as an Annotated TypedDict instead
class Facts(TypedDict):
    fact_1: Annotated[str, "Fact 1 about the topic"]
    fact_2: Annotated[str, "Fact 2 about the topic"]
    fact_3: Annotated[str, "Fact 3 about the topic"]


structured_model = model.with_structured_output(Facts)


# Prompt -- no format_instruction needed at all: with_structured_output
# communicates the schema to the model directly, so there's nothing to
# generate and splice into the template the way JsonOutputParser and
# StructuredOutputParser both required.
template = PromptTemplate(
    template="Give 3 facts about {topic}",
    input_variables=["topic"],
)


# Chain
chain = template | structured_model


try:
    result = chain.invoke({
        "topic": "black hole"
    })

    print(result)

except Exception as e:
    print(f"Something went wrong getting structured facts: {e}")