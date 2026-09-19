
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
import os

from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()



#SENTIMENT MODEL
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)



#LLM MODEL - GEMMA
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-12b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

model = ChatHuggingFace(llm=llm)



# STRUCTURED OUTPUT SCHEMA

class Review(TypedDict):
    key_themes: Annotated[list[str], "The key themes of the review."]
    summary: Annotated[str, "A concise summary of the review."]
    sentiment: Annotated[str, "The sentiment returned by the sentiment classification model."]
    pros: Annotated[Optional[list[str]], "The pros mentioned in the review."]
    cons: Annotated[Optional[list[str]], "The cons mentioned in the review."]

structured_model = model.with_structured_output(Review)



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