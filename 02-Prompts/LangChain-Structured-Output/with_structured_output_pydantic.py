from dotenv import load_dotenv
from typing import Literal, Optional
import os
import json

from pydantic import BaseModel, Field

from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# ---------------------------------------------------------------------------
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# This file demonstrates how to combine TWO specialized models:
#
#   1. Cardiff NLP sentiment classifier
#      -> determines the sentiment of the review.
#
#   2. Gemma 3
#      -> extracts themes, summary, pros, cons, and reviewer name.
#
# We then use Pydantic to validate the structured result returned by Gemma.
#
# IMPORTANT:
# HuggingFaceEndpoint + the current inference provider being used here does
# not support the function-calling mechanism required by:
#
#     model.with_structured_output(Review)
#
# Therefore, we cannot use native Pydantic structured output in this setup.
#
# Instead, we use:
#
#     Gemma
#       ↓
#     JSON text
#       ↓
#     json.loads()
#       ↓
#     Review.model_validate()
#       ↓
#     validated Pydantic object
#
# This is prompt-based structured output followed by programmatic validation.
# ---------------------------------------------------------------------------


load_dotenv()


# ---------------------------------------------------------------------------
# SENTIMENT MODEL
# ---------------------------------------------------------------------------
# A dedicated sentiment classifier handles ONLY sentiment.
#
# This is preferable here because sentiment is a narrow classification task,
# while Gemma is being used for the broader language-understanding tasks.
# ---------------------------------------------------------------------------

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


# ---------------------------------------------------------------------------
# LLM MODEL - GEMMA
# ---------------------------------------------------------------------------
# Gemma handles:
#   - key themes
#   - summary
#   - pros
#   - cons
#   - reviewer name
#
# The Hugging Face token is loaded from .env.
# ---------------------------------------------------------------------------

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-12b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=(
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    ),
)

model = ChatHuggingFace(llm=llm)


# ---------------------------------------------------------------------------
# STRUCTURED OUTPUT SCHEMA
# ---------------------------------------------------------------------------
# Pydantic gives us a formal structure for the information we expect.
#
# IMPORTANT:
# Pydantic does NOT force Gemma to generate this structure.
# It validates the result AFTER Gemma responds.
# ---------------------------------------------------------------------------

class Review(BaseModel):

    key_themes: list[str] = Field(
        description="The key themes of the review."
    )

    summary: str = Field(
        description="A concise summary of the review."
    )

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment returned by the sentiment classification model."
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="The pros mentioned in the review."
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="The cons mentioned in the review."
    )

    name: Optional[str] = Field(
        default=None,
        description="The name of the reviewer."
    )


# ---------------------------------------------------------------------------
# REVIEW
# ---------------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # 1. GET SENTIMENT FROM CARDIFF NLP MODEL
    # -----------------------------------------------------------------------
    # The dedicated classifier determines the sentiment.
    #
    # Example output:
    #
    # {
    #     "label": "Positive",
    #     "score": 0.98
    # }
    #
    # We normalize the label to lowercase once so that it matches the
    # Literal values defined in our Pydantic schema.
    # -----------------------------------------------------------------------

    sentiment_result = sentiment_model(review)[0]

    sentiment = sentiment_result["label"].lower()

    print("Sentiment:", sentiment)


    # -----------------------------------------------------------------------
    # 2. ASK GEMMA TO RETURN JSON
    # -----------------------------------------------------------------------
    # We cannot use:
    #
    #     model.with_structured_output(Review)
    #
    # because the current Hugging Face setup does not support the required
    # function-calling mechanism.
    #
    # Therefore, we explicitly instruct Gemma to return JSON.
    # -----------------------------------------------------------------------

    response = model.invoke(
        f"""
Analyze the following product review.

Review:
{review}

The sentiment has already been determined by a dedicated
sentiment classification model.

Sentiment from classification model:
{sentiment}

Use this sentiment exactly as provided.
Do not independently classify the sentiment.

Extract:

- key_themes: the main themes discussed in the review
- summary: a concise summary covering the important positives and negatives
- sentiment: use exactly "{sentiment}"
- pros: the positive aspects mentioned in the review
- cons: the negative aspects mentioned in the review
- name: the reviewer's name, if available

Return ONLY valid JSON.

Do not use Markdown.
Do not use ```json.
Do not add any explanation before or after the JSON.

The JSON must follow exactly this structure:

{{
    "key_themes": ["theme 1", "theme 2"],
    "summary": "A concise summary",
    "sentiment": "{sentiment}",
    "pros": ["positive aspect 1", "positive aspect 2"],
    "cons": ["negative aspect 1", "negative aspect 2"],
    "name": "Muaviz Khan"
}}
"""
    )


    # -----------------------------------------------------------------------
    # 3. EXTRACT THE MODEL'S TEXT
    # -----------------------------------------------------------------------
    # ChatHuggingFace normally returns an AIMessage.
    # Its actual generated text is stored in .content.
    # -----------------------------------------------------------------------

    response_text = response.content.strip()

    print("\nRAW GEMMA RESPONSE:")
    print(response_text)


    # -----------------------------------------------------------------------
    # 4. CONVERT JSON TEXT → PYTHON DICTIONARY
    # -----------------------------------------------------------------------
    # json.loads() parses the text generated by Gemma.
    #
    # If Gemma produces invalid JSON, this step raises JSONDecodeError.
    # -----------------------------------------------------------------------

    parsed_result = json.loads(response_text)


    # -----------------------------------------------------------------------
    # 5. VALIDATE THE RESULT WITH PYDANTIC
    # -----------------------------------------------------------------------
    # This is the important difference:
    #
    # Gemma generates the structure.
    # Pydantic verifies the structure.
    #
    # If fields are missing or have incompatible values, Pydantic raises
    # a validation error.
    # -----------------------------------------------------------------------

    validated_review = Review.model_validate(parsed_result)


    # -----------------------------------------------------------------------
    # 6. COMBINE THE RESULTS
    # -----------------------------------------------------------------------
    # We deliberately take sentiment from the dedicated classifier rather
    # than trusting Gemma's sentiment field.
    #
    # This guarantees that the final sentiment comes from the specialized
    # sentiment model.
    # -----------------------------------------------------------------------

    result = {
        "key_themes": validated_review.key_themes,
        "summary": validated_review.summary,
        "sentiment": sentiment,
        "pros": validated_review.pros,
        "cons": validated_review.cons,
        "name": validated_review.name
    }


    # -----------------------------------------------------------------------
    # 7. FINAL RESULT
    # -----------------------------------------------------------------------

    print("\nFINAL STRUCTURED RESULT:")
    print(result)


except json.JSONDecodeError as e:

    print("\nGemma returned invalid JSON.")
    print("JSON parsing error:", e)


except Exception as e:

    print(f"\nSomething went wrong extracting the structured review: {e}")