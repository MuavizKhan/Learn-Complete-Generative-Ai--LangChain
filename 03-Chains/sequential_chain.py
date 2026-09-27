# ---------------------------------------------------------------------------
# sequential_chain.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# stroutputparser.py did "report, then summarize" as TWO separate chains,
# with you manually feeding chain1's result into chain2.invoke({"text": ...})
# yourself. This file does the exact same job as ONE chain:
#
#   prompt1 | model | parser | prompt2 | model | parser
#
# One .invoke() call runs all six stages. That's what "Sequential Chain"
# means here -- multiple prompt-model-parser stages linked directly
# together, instead of you wiring separate chains together by hand.
#
# WHY does the plain string coming out of the first parser correctly land
# in prompt2's {text} slot, with no dict, no key name anywhere? I tested
# this directly rather than assume: a PromptTemplate with exactly ONE
# input variable will accept a bare string and use it for that one
# variable automatically. That's the trick making this whole chain click
# together as one straight line. It only works because prompt2 needs
# exactly one variable -- a template needing two or more couldn't accept a
# single string this way, which is exactly the situation Parallel Chain,
# next file, is built to handle.
# ---------------------------------------------------------------------------

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=['text']
)

# Same truncation risk as stroutputparser.py's "detailed report" prompt,
# and for the same reason: this model handles BOTH stages here, so one
# shared max_new_tokens has to be generous enough for the longer of the
# two (the report), even though the summary stage barely needs it.
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

try:
    result = chain.invoke({'topic': 'GDP in India'})
    print(result)

    chain.get_graph().print_ascii()

except Exception as e:
    print(f"Something went wrong running the chain: {e}")

