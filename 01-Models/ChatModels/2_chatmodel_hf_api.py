# ---------------------------------------------------------------------------
# 2_chatmodel_hf_api.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# The free, Hugging Face version of the same "Chat Model" idea from
# 1_chatmodel_openai.py. Notice it builds TWO objects instead of one:
#
#   HuggingFaceEndpoint -> the raw connection to a model hosted on
#                           Hugging Face's free Inference API. On its own,
#                           this behaves like a plain completion model
#                           (text in, text out) -- the "LLM" flavor, not
#                           the "Chat Model" flavor.
#
#   ChatHuggingFace      -> wraps that raw connection and adds the actual
#                           chat behaviour: it formats your messages using
#                           the exact chat template Llama 3.1 was trained
#                           on (its own special role tokens), so the model
#                           responds the way an instruction-tuned chat
#                           model is supposed to.
#
# OpenAI bundles both of these jobs into one ChatOpenAI class. Open-source
# models on Hugging Face keep them separate, which is why this file has two
# objects where 1_chatmodel_openai.py only needed one.
# ---------------------------------------------------------------------------

import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise EnvironmentError(
        "HUGGINGFACEHUB_API_TOKEN is missing. Add it to your .env file.\n"
        "Get a free token at https://huggingface.co/settings/tokens"
    )

# The raw connection to the model on Hugging Face's Inference API.
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2,       # factual question -> low temperature, per your notes
    max_new_tokens=256,    # a deliberate cap, instead of relying on the default
)

# Wraps the raw connection so it behaves like a proper Chat Model.
model = ChatHuggingFace(llm=llm)

try:
    result = model.invoke("What is the capital of France?")
    # This IS a Chat Model, so .invoke() returns a full AIMessage object --
    # same as ChatOpenAI in the last file. print(result) alone dumps the
    # whole object; .content is the clean reply text.
    print(result.content)
except Exception as e:
    print(f"Something went wrong calling the model: {e}")