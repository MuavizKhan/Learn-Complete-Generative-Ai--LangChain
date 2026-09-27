# ---------------------------------------------------------------------------
# simple_chain.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# Nothing new about the chain itself -- prompt | model | parser is exactly
# the shape stroutputparser.py already built, twice over, in the last
# folder. This file's actual job is naming that shape formally: this is a
# "Simple Chain," the first and most basic of the four chain types in this
# module (Simple -> Sequential -> Parallel -> Conditional, one file each).
#
# The one genuinely new line is the last one: chain.get_graph().print_ascii()
# draws the chain's actual structure as text, without calling the model at
# all -- pure introspection. Worth keeping in your back pocket once chains
# get more complex a couple of files from now.
# ---------------------------------------------------------------------------

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser

try:
    result = chain.invoke({'topic': 'cricket'})
    print(result)

    # NOTE: this line needs one extra package that isn't installed by
    # default -- I hit this myself testing it: `pip install grandalf`.
    # Without it, this raises ImportError rather than printing anything.
    # Another one for that still-empty requirements.txt from lesson one.
    chain.get_graph().print_ascii()

except Exception as e:
    print(f"Something went wrong running the chain: {e}")