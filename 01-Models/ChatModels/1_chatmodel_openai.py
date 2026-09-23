# ---------------------------------------------------------------------------
# 1_chatmodel_openai.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# This is LangChain's "Chat Model" interface -- the modern way almost every
# provider exposes their models today. Compare this to 01-Models/LLMs, where
# a plain string went in and a plain string came out. Here, a Chat Model
# always hands back a *message object*, not raw text -- watch the last line.
#
# This file specifically talks to OpenAI. The next two files in this same
# folder (2_chatmodel_hf_api.py and 3_chatmodel_hf_local.py) show the exact
# same idea for free, using Hugging Face -- this one exists so you can see
# what a paid, closed-source provider's interface looks like, side by side.
# Running this file DOES spend real OpenAI credits (a cheap model, but not
# a free one) -- unlike the other two files in this folder.
# ---------------------------------------------------------------------------

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Reads your .env file so OPENAI_API_KEY is available below without ever
# being typed into this file directly.
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY is missing. Add it to your .env file.\n"
        "Get one at https://platform.openai.com/api-keys"
    )

# model       -> which OpenAI model to call. gpt-5.6-luna is OpenAI's
#                fastest, cheapest current tier -- a sensible, low-cost
#                default for a simple factual question like this one.
# temperature -> this prompt has exactly one correct answer, so we want the
#                model playing it safe, not creative. Your own notes put
#                factual answers at 0.0-0.3 -- we'll use 0.2.
model = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0.2,
)

try:
    result = model.invoke("What is the Capital of India?")
    # Chat Models always hand back a message object, never a plain string --
    # the actual reply text lives inside .content.
    print(result.content)
except Exception as e:
    print(f"Something went wrong calling the model: {e}")