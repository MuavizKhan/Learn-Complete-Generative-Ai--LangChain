# ---------------------------------------------------------------------------
# messages.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# Every ChatModel call so far has passed a single plain string to .invoke().
# LangChain was quietly wrapping that string into a HumanMessage for you
# the whole time. This file stops relying on that and builds the message
# list by hand instead -- which is the only way to add a SystemMessage
# (instructions for how the model should behave) or to give it more than
# one turn of context.
#
# It also demonstrates the actual fix to a problem raised all the way back
# in the very first lesson: an LLM call has no memory between requests.
# The last line of this file is that fix in practice -- manually appending
# the model's own reply back into the list, so the NEXT call could include
# it.
# ---------------------------------------------------------------------------

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# A note on the model choice: gpt-oss-120b is OpenAI's own open-weight
# model (Apache 2.0, released August 2025) -- a genuinely capable, current
# choice, not a random pick. But at 120 billion parameters it's a different
# weight class from Llama-3.1-8B (used earlier in this repo): it's too
# large for Hugging Face's free serverless tier on its own, so calling it
# here will most likely draw on paid Hugging Face inference credits rather
# than being fully free -- typically a fraction of a cent per call, not
# nothing. If you want a guaranteed-free swap that needs no code changes
# beyond this one line, meta-llama/Llama-3.1-8B-Instruct (already used
# elsewhere in this repo) is a safe substitute:
# repo_id="meta-llama/Llama-3.1-8B-Instruct"
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)

# Three roles, three classes:
#   SystemMessage -> instructions for the model itself, not something
#                    either person said. Sets behavior for the WHOLE
#                    conversation, once, up front.
#   HumanMessage  -> something the user said.
#   AIMessage     -> something the model said. Used below to record the
#                    model's own reply back into this same list.
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain?")
]

try:
    # .invoke() also accepts a LIST of Message objects, not just a plain
    # string -- this is what actually gives you control over roles.
    result = model.invoke(messages)

    # This line is the whole point of the file: without it, the next call
    # would know nothing about this one. Appending the reply, as an
    # AIMessage, is what turns a single question-and-answer into the start
    # of an actual conversation with memory.
    messages.append(AIMessage(content=result.content))

    # Prints the full list of typed objects (each with its role attached),
    # not just plain text -- useful for seeing that this really is a
    # structured object, not a chat log string. If you just want the
    # readable conversation, loop over messages and print(m.content) for
    # each instead.
    print(messages)

except Exception as e:
    print(f"Something went wrong calling the model: {e}")