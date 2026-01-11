import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

class HuggingFaceLLM:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id

    def build(self) -> ChatHuggingFace:
        # Check for API token
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            print("Warning: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")

        llm = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            task="conversational",  # Required for instruction tuned models via API
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        )
        return ChatHuggingFace(llm=llm)
