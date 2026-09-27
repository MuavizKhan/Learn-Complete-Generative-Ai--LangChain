from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
import os

from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ---------------------------------------------------------------------------
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# Two different models, each doing the ONE job it's actually best at,
# instead of asking a single general-purpose LLM to do everything:
#   - a small, purpose-built classifier (cardiffnlp/twitter-roberta-base-
#     sentiment-latest) handles sentiment -- narrower job, and a dedicated
#     classifier is typically more consistent at it than a general LLM
#     improvising the same judgment.
#   - Gemma 3 (a general LLM) handles everything that actually benefits
#     from language understanding: pulling out themes, summarizing, and
#     splitting pros from cons.
# The LLM is then explicitly told the sentiment is already decided and not
# to re-guess it -- so the two models' jobs don't overlap or conflict.
#
# The other new piece: model.with_structured_output(Review) is what makes
# this whole thing possible. It's the difference between "hope the model's
# reply parses as JSON" and telling the model provider directly "your
# response must match this exact shape" -- Review, the Annotated TypedDict
# below, is that shape.
# ---------------------------------------------------------------------------

load_dotenv()


# SENTIMENT MODEL
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


# LLM MODEL - GEMMA
# Note on the token: this file explicitly reads HF_TOKEN, which is
# genuinely the current, recommended env var name -- but every earlier
# file in this repo relied on HuggingFaceEndpoint's automatic fallback,
# which historically meant HUGGINGFACEHUB_API_TOKEN. I checked the
# installed library's actual source rather than assume: its own internal
# fallback ONLY checks HF_TOKEN, not the older name. If your .env file
# only has the legacy name set, this line alone won't find it. Checking
# both here means it works either way, regardless of which one is in
# your .env.
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-12b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)


# STRUCTURED OUTPUT SCHEMA
# Annotated[type, "description"] -- the upgrade over last file's bare
# TypedDict. A plain `summary: str` tells the model there's a string; the
# description text alongside it tells the model WHAT to put there. Without
# these descriptions, with_structured_output only enforces shape, not intent.
class Review(TypedDict):
    key_themes: Annotated[list[str], "The key themes of the review."]
    summary: Annotated[str, "A concise summary of the review."]
    sentiment: Annotated[str, "The sentiment returned by the sentiment classification model."]
    pros: Annotated[Optional[list[str]], "The pros mentioned in the review."]
    cons: Annotated[Optional[list[str]], "The cons mentioned in the review."]

structured_model = model.with_structured_output(Review)


# REVIEW
review = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Muaviz Khan
"""

try:
    # 5. GET SENTIMENT FROM CARDIFF NLP MODEL
    sentiment_result = sentiment_model(review)[0]
    sentiment = sentiment_result["label"]

    # 6. GET SUMMARY USING GEMMA
    summary_result = structured_model.invoke(
        f"""
Analyze the following product review.

Review:
{review}

The sentiment has already been determined by the dedicated
sentiment classification model.

Sentiment from classification model:
{sentiment}

Use this sentiment exactly as provided.
Do not independently classify the sentiment.

Extract the following information from the review:

- key_themes: Identify the main themes discussed in the review.
- summary: Provide a concise summary covering the main positives and negatives.
- sentiment: Use exactly "{sentiment}".
- pros: List the positive aspects of the product mentioned in the review.
- cons: List the negative aspects of the product mentioned in the review.

Return the result using the required structured output.
"""
    )

    # COMBINE THE RESULTS
    result = {
        "key_themes": summary_result["key_themes"],
        "summary": summary_result["summary"],
        "sentiment": sentiment,
        "pros": summary_result["pros"],
        "cons": summary_result["cons"]
    }

    print(result)

except Exception as e:
    print(f"Something went wrong extracting the structured review: {e}")