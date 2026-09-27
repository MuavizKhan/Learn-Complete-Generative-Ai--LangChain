# ---------------------------------------------------------------------------
# chat_prompt_template.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# Last file (messages.py) built a fixed, hardcoded list of Message objects.
# This is the templated version of that same idea: instead of writing out
# "You are a helpful assistant" every time, you write "You are a helpful
# {domain} expert" ONCE and fill in the blank differently on every call.
#
# This file deliberately stops short of actually calling a model. It only
# builds the prompt and shows what comes out the other end -- a cheap,
# free way to check a template is filling in correctly BEFORE spending a
# real model call on it. Nothing is missing here; that's the whole scope.
# ---------------------------------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate

# Each tuple is (role, template string). Note this is NOT SystemMessage(...)
# or HumanMessage(...) like last file -- those are objects with fixed text
# already inside them. This is a template: the {domain} and {topic}
# placeholders don't become real message objects until .invoke() below
# fills them in.
chat_template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful {domain} expert."),
    ('human', "Explain in simple terms, what is {topic}?")
])

try:
    # .invoke() here isn't calling a model -- it's "running" the template
    # itself, filling {domain} and {topic} in from this dictionary. Same
    # method name as model.invoke() from every earlier file, doing the
    # equivalent job for a completely different kind of object: this is
    # LangChain's Runnable pattern, and it's everywhere.
    prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'dusra'})

    # prompt is NOT the list of messages -- it's a small wrapper object
    # (a ChatPromptValue) holding the filled-in messages. It exists so the
    # exact same template can feed either a Chat Model (via prompt.messages
    # or prompt.to_messages(), a list of Message objects) or a plain LLM
    # (via prompt.to_string(), one flattened string) -- one template,
    # two possible destinations.
    print(prompt.messages)

except Exception as e:
    print(f"Something went wrong building the prompt: {e}")