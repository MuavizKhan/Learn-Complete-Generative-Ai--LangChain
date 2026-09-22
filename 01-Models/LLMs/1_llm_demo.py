# ---------------------------------------------------------------------------
# 1_llm_demo.py  (Hugging Face edition -- free to run)
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# This file demonstrates the basic "LLM" abstraction in LangChain.
#
# The idea is simple:
#
#     Plain String  →  LLM  →  Plain String
#
# We give the model one plain text prompt and get one plain text response.
#
# This is different from a "Chat Model", where we work with messages having
# roles such as system, human, and AI.
#
# WHY ARE WE USING HuggingFacePipeline?
# -------------------------------------
# We want to run the model locally instead of calling a paid/remote API.
#
# HuggingFacePipeline allows a Hugging Face Transformers model to be used
# through LangChain's LLM interface.
#
# This means we can learn the LangChain LLM abstraction without depending
# on an external inference provider.
# ---------------------------------------------------------------------------


import os
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# STEP 1: Load environment variables
# ---------------------------------------------------------------------------
#
# WHY?
# ----
# We use load_dotenv() because this project also contains examples that use
# Hugging Face API tokens stored inside a .env file.
#
# The model in THIS file runs locally, so an API token is not required here.
# We keep load_dotenv() because the project follows the same environment
# variable setup across its different examples.
#
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# STEP 2: Create the Hugging Face model pipeline
# ---------------------------------------------------------------------------
#
# WHY?
# ----
# LangChain itself does not contain the actual language model.
#
# We first need to load a model using Hugging Face Transformers.
#
# The pipeline() function gives us a simple interface for working with the
# model:
#
#     prompt → model → generated text
#
# We use an instruction-tuned model because we want the model to understand
# prompts such as:
#
#     "Write a short poem about the beauty of nature."
#
# Qwen2.5-0.5B-Instruct is relatively small, so it is suitable for learning
# and local experimentation without downloading a very large model.
#
# "text-generation"
# ------------------
# Tells Transformers that the model should generate text based on the
# supplied prompt.
#
# max_new_tokens
# --------------
# Controls how many new tokens the model is allowed to generate.
# It prevents the model from generating an unnecessarily long response.
#
# ---------------------------------------------------------------------------

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=256,
)


# ---------------------------------------------------------------------------
# STEP 3: Connect the Hugging Face pipeline to LangChain
# ---------------------------------------------------------------------------
#
# WHY?
# ----
# At this point, we have a Hugging Face model, but we want to use it through
# LangChain.
#
# HuggingFacePipeline acts as the bridge:
#
#     Hugging Face Model
#            ↓
#     HuggingFacePipeline
#            ↓
#       LangChain LLM
#
# This is useful because LangChain provides a common interface such as
# .invoke(), which can later be used with prompts, chains, runnables,
# output parsers, retrievers, etc.
#
# ---------------------------------------------------------------------------

llm = HuggingFacePipeline(
    pipeline=generator
)


# ---------------------------------------------------------------------------
# STEP 4: Send a prompt to the LLM
# ---------------------------------------------------------------------------
#
# WHY?
# ----
# The key idea of an LLM is that we can provide a text prompt and ask the
# model to generate a continuation/response.
#
# Because this example demonstrates the LLM abstraction, we provide ONE
# plain string.
#
# We are NOT creating:
#
#     SystemMessage(...)
#     HumanMessage(...)
#     AIMessage(...)
#
# That belongs to the Chat Model abstraction.
#
# .invoke()
# ---------
# .invoke() is LangChain's standard method for executing a runnable/model.
#
# Here:
#
#     String → LLM → String
#
# ---------------------------------------------------------------------------

try:

    result = llm.invoke(
        "Write a short poem about the beauty of nature."
    )

    print(result)


except Exception as e:

    # WHY?
    # ----
    # If the model fails because of an installation, model-loading, or
    # generation problem, we want a readable error instead of hiding
    # what went wrong.

    print("Something went wrong calling the model:")
    print(type(e).__name__)
    print(repr(e))