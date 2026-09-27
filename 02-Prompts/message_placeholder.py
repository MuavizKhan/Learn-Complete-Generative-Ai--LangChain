# ---------------------------------------------------------------------------
# message_placeholder.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# chat_prompt_template.py filled a SINGLE placeholder ({topic}) with a
# single string. MessagesPlaceholder is different: it's a slot that gets
# filled with an entire LIST of prior messages at once -- this is how you
# hand a template "everything said so far," on top of new templated
# content, in one combined prompt.
# ---------------------------------------------------------------------------

import re
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful customer support assistant.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# WHY not just chat_history.extend(f.readlines()), like the original?
# chat_history.txt stores lines that LOOK like Python --
# HumanMessage(content="...") -- but a text file only ever gives you
# characters, never runs them as code. f.readlines() would have handed
# MessagesPlaceholder a list of two identical-looking PLAIN STRINGS, with
# no actual HumanMessage/AIMessage objects anywhere.
#
# I ran exactly that version to see what actually happens, since guessing
# felt too easy to get wrong: it doesn't crash. LangChain's message
# coercion falls back to wrapping any plain string as a HumanMessage --
# the same auto-wrapping behind model.invoke("a string") several files
# ago. The result: BOTH lines silently became HumanMessages, one of them
# literally containing the text "AIMessage(content=..." as if the
# customer had typed it. Wrong role, garbled content, zero errors raised
# -- a quiet failure is worse than a loud one, which is exactly why this
# was worth actually running rather than assuming.
#
# The fix: parse each line properly and build the real objects.
MESSAGE_LINE = re.compile(r'^(HumanMessage|AIMessage)\(content="(.*)"\)$')

def load_chat_history(path: str) -> list:
    history = []
    with open(path) as f:
        for line in f:
            match = MESSAGE_LINE.match(line.strip())
            if not match:
                continue  # skip anything that doesn't match the expected shape
            msg_type, content = match.groups()
            history.append(
                HumanMessage(content=content) if msg_type == "HumanMessage"
                else AIMessage(content=content)
            )
    return history

try:
    chat_history = load_chat_history('chat_history.txt')

    print(chat_history)

    # create our prompt
    prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'What is the status of my order?'})

    print(prompt)

except Exception as e:
    print(f"Something went wrong building the prompt: {e}")