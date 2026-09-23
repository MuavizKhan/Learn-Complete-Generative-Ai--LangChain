# ---------------------------------------------------------------------------
# 3_chatmodel_hf_local.py
#
# WHAT THIS FILE DEMONSTRATES
# ---------------------------------------------------------------------------
# The third way to get a Chat Model in LangChain: running the model
# yourself, on your own machine, instead of calling anyone's API.
#
#   HuggingFacePipeline -> downloads the model's weights from Hugging Face
#                          ONCE (cached after that) and runs inference
#                          locally, using the `transformers` library under
#                          the hood. No network call happens at generation
#                          time -- which is also why this file needs no
#                          token and no load_dotenv() at all. That's
#                          correct, not a bug: nothing secret is involved
#                          when everything runs on your own hardware.
#
#   ChatHuggingFace      -> exactly the same wrapping job as in
#                          2_chatmodel_hf_api.py -- adds chat-template
#                          formatting on top of a raw text-generation
#                          backend.
#
# Notice the model is different too: TinyLlama-1.1B-Chat instead of
# Llama-3.1-8B-Instruct from the last file. That's deliberate, not an
# accidental downgrade -- an 8B model needs serious GPU memory to run
# locally at a reasonable speed; a 1.1B model can run on an ordinary
# laptop CPU, just slower than a hosted API would be.
# ---------------------------------------------------------------------------

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    # NOTE: generation settings for a local pipeline are passed as a nested
    # dict here, not as direct keyword arguments the way HuggingFaceEndpoint
    # took them in the last file -- an API difference worth remembering.
    pipeline_kwargs={
        "max_new_tokens": 100,
        # Lowered from 0.7 to 0.2: this is a factual question with exactly
        # one right answer, so per your own notes (0.0-0.3 for factual
        # answers) we want the model boring, not creative. Small local
        # models like this 1.1B one also tend to wander off-topic more
        # easily at higher temperatures than a large hosted model would.
        "temperature": 0.2,
    },
    # Uncomment if you have a GPU available (e.g. a Colab GPU runtime) --
    # it noticeably speeds up local generation. Needs one extra package:
    # pip install accelerate
    # device_map="auto",
)

model = ChatHuggingFace(llm=llm)

try:
    result = model.invoke("What is the capital of France?")
    print(result.content)
except Exception as e:
    print(f"Something went wrong calling the model: {e}")