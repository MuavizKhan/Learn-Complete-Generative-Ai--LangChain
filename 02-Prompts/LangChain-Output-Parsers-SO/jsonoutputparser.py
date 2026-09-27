# ---------------------------------------------------------------------------
# jsonoutputparser.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# JsonOutputParser does two jobs, not one: it can TELL a model how to format
# its reply (parser.get_format_instructions()), and it PARSES that reply
# back into a dict once it comes in. The previous folder's structured-
# output files hand-wrote "Return ONLY valid JSON..." themselves; this is
# LangChain's own built-in version of that same instruction.
#
# No pydantic_object is given to JsonOutputParser here, which matters: it
# guarantees the reply IS valid JSON, but not what shape that JSON takes --
# nothing tells the model what keys to use for "5 facts." Compare that to
# the previous folder, where a schema pinned down the exact structure.
# These two ideas -- constraining shape (Structured Output) vs. parsing
# whatever comes back (Output Parsers) -- are related but genuinely
# different tools.
# ---------------------------------------------------------------------------

from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

# WHY partial_variables, and not just another {curly brace}?
# input_variables (topic) get filled later, at .invoke() time -- one value
# per call. partial_variables get filled ONCE, right now, when this
# template object is built. get_format_instructions() only needs to run
# once, so it belongs here, not as something you'd re-supply on every call.
#
# I resolved this exact template to check what the model actually receives:
#   "Give me 5 facts about black hole
#    Return a JSON object."
# That's it -- get_format_instructions() is genuinely just that one line
# when no schema is attached, which is exactly why the output's shape isn't
# guaranteed here.
template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

try:
    # Worth knowing: JsonOutputParser can fail here in a way StrOutputParser
    # never could -- if the model doesn't return something that parses as
    # JSON, this raises an OutputParserException. It IS more forgiving than
    # a raw json.loads() would be, though -- it strips markdown code fences
    # automatically. I tested this directly: text wrapped in a ```json
    # fence parsed correctly with no extra code at all, the same fence-
    # stripping the previous folder had to write by hand.
    result = chain.invoke({'topic': 'black hole'})
    print(result)

except Exception as e:
    print(f"Something went wrong parsing the model's output as JSON: {e}")