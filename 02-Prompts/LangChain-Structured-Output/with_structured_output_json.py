from dotenv import load_dotenv
from typing import Literal, Optional
import os

from pydantic import BaseModel

from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ---------------------------------------------------------------------------
# WHAT CHANGED FROM THE ORIGINAL FILE, AND WHY
# ---------------------------------------------------------------------------
# This file defined a raw JSON Schema dict (json_schema) AND a Pydantic
# class (Review) -- but only ever used the Pydantic class, and only for a
# manual validation step at the very end. json_schema itself was never
# passed anywhere; with_structured_output() was never called at all. This
# version actually uses json_schema the way the filename promises:
#   structured_model = model.with_structured_output(json_schema)
# That's also the real, useful difference this file is supposed to teach
# versus the last two: pass a Pydantic class, and LangChain hands back a
# validated Pydantic object automatically. Pass a raw JSON Schema dict (no
# Python class behind it), and LangChain can't validate anything for you --
# it hands back a plain, unvalidated dict instead. Confirmed directly from
# with_structured_output's own docstring: "If schema is a Pydantic class...
# validated... Otherwise the model output will be a dict and will not be
# validated." That's the actual trade-off between these three files, not
# just three different ways of writing the same schema.
#
# Because a raw JSON Schema gives up automatic validation, re-validating
# the result with the Review Pydantic class afterward (kept below, used
# properly this time) is a reasonable safety net if you still want one --
# optional, not required, and worth knowing it's optional.
#
# The schema itself also had a real bug: sentiment's enum only listed
# ["pos", "neg"] -- two abbreviated values, while its own description said
# "negative, positive or neutral" and every other file in this folder uses
# the full, unabbreviated words. "Neutral" wasn't a valid option at all,
# so a genuinely neutral review had no correct value to be given. Fixed to
# match what the rest of the pipeline actually uses.
# ---------------------------------------------------------------------------

load_dotenv()


# SENTIMENT MODEL
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


# LLM MODEL - GEMMA
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-12b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)


# STRUCTURED OUTPUT SCHEMA -- a raw JSON Schema dict, not a Python class
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["positive", "negative", "neutral"],
      "description": "The sentiment of the review: positive, negative, or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {"type": "string"},
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {"type": "string"},
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

# Optional safety net: since json_schema alone gives no validation, this
# Pydantic class lets us double-check the dict we get back, the same shape
# as json_schema above, expressed as an actual Python class.
class Review(BaseModel):
    key_themes: list[str]
    summary: str
    sentiment: Literal["positive", "negative", "neutral"]
    pros: Optional[list[str]] = None
    cons: Optional[list[str]] = None
    name: Optional[str] = None


structured_model = model.with_structured_output(json_schema)


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
    # GET SENTIMENT FROM CARDIFF NLP MODEL
    sentiment_result = sentiment_model(review)[0]
    sentiment = sentiment_result["label"].lower()  # normalize once, at the source

    # GET STRUCTURED RESULT USING GEMMA -- one call, no manual JSON parsing
    raw_result = structured_model.invoke(
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

Extract the key themes, a concise summary, the pros, the cons,
and the reviewer's name if one is given.
"""
    )

    # raw_result is a plain dict here (no Python class behind json_schema),
    # so unlike the last two files, nothing validated it automatically.
    # Running it through Review confirms it actually matches the shape
    # before we trust it.
    validated_review = Review.model_validate(raw_result)

    # COMBINE THE RESULTS
    result = {
        "key_themes": validated_review.key_themes,
        "summary": validated_review.summary,
        "sentiment": sentiment,
        "pros": validated_review.pros,
        "cons": validated_review.cons,
        "name": validated_review.name
    }

    print("\nFINAL STRUCTURED RESULT:")
    print(result)

except Exception as e:
    print(f"Something went wrong extracting the structured review: {e}")