# ---------------------------------------------------------------------------
# chatbot.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# Everything from this folder so far, combined into something you can
# actually talk to: SystemMessage/HumanMessage/AIMessage and the
# append-for-memory pattern from messages.py, now wrapped in a loop so the
# conversation keeps growing turn by turn instead of running once and
# stopping.
# ---------------------------------------------------------------------------

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

# Same model and cost note as messages.py -- but it matters more here.
# Every turn re-sends the ENTIRE chat_history, not just the newest message
# (a chat model has no memory of its own between calls -- that's the whole
# reason this list exists), so a long conversation costs progressively
# more per turn, not a flat amount per question. Swap in
# meta-llama/Llama-3.1-8B-Instruct below for a guaranteed-free version
# while you're experimenting.
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.8,
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    user_input = input('You: ').strip()

    # Checking for exit BEFORE touching chat_history at all: the original
    # order appended "exit" first and checked second, so every saved
    # conversation ended with a question the AI never got to answer. I ran
    # the original ordering with fake input to see the actual result:
    #   HumanMessage -> 'exit'      <- last entry, nothing replies to it
    # Matching a few spellings and ignoring case too, since people don't
    # reliably type "exit" exactly.
    if user_input.lower() in ('exit', 'quit', 'bye'):
        break

    chat_history.append(HumanMessage(content=user_input))

    try:
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        print("AI:", result.content)
    except Exception as e:
        # One failed call shouldn't end the whole conversation -- but leaving
        # this turn's HumanMessage in place with no reply would let a retry
        # add a second HumanMessage right after it, with no AI turn between
        # them. Removing it keeps the history consistent for the next try.
        chat_history.pop()
        print(f"Something went wrong calling the model: {e}")

print(chat_history)