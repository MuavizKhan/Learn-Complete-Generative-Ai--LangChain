from dotenv import load_dotenv
from typing import Literal, Optional
import os
import json

from pydantic import BaseModel, Field

from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


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
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

model = ChatHuggingFace(llm=llm)


# STRUCTURED OUTPUT SCHEMA
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
        description="The pros mentioned in the review."
    )

    cons: Optional[list[str]] = Field(
        description="The cons mentioned in the review."
    )

    name: Optional[str] = Field(
        default=None,
        description="The name of the reviewer."
    )


# REVIEW
review = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Muaviz Khan
"""


# GET SENTIMENT FROM CARDIFF NLP MODEL
sentiment_result = sentiment_model(review)[0]

sentiment = sentiment_result["label"]


# GET SUMMARY USING GEMMA
summary_result = model.invoke(
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

Extract the following information:

- key_themes: Identify the main themes discussed in the review.
- summary: Provide a concise summary covering the main positives and negatives.
- sentiment: Use exactly "{sentiment}".
- pros: List the positive aspects of the product mentioned in the review.
- cons: List the negative aspects of the product mentioned in the review.
- name: Extract the reviewer's name if available.

Return ONLY valid JSON.

The JSON must follow this exact structure:

{{
    "key_themes": ["theme 1", "theme 2"],
    "summary": "concise summary",
    "sentiment": "{sentiment}",
    "pros": ["pro 1", "pro 2"],
    "cons": ["con 1", "con 2"],
    "name": "reviewer name"
}}
"""
)


# GET GEMMA'S RAW RESPONSE
# GET GEMMA'S RAW RESPONSE
raw_output = summary_result.content

print("\nRAW GEMMA OUTPUT:")
print(raw_output)


# REMOVE MARKDOWN CODE FENCES
raw_output = raw_output.strip()

if raw_output.startswith("```json"):
    raw_output = raw_output[7:-3].strip()
elif raw_output.startswith("```"):
    raw_output = raw_output[3:-3].strip()


# CONVERT JSON STRING TO PYTHON DICTIONARY
parsed_output = json.loads(raw_output)


# VALIDATE USING PYDANTIC
validated_review = Review.model_validate(parsed_output)


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
