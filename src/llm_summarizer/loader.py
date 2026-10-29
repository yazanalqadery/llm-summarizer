import json
from llm_summarizer.models import Document


def load_docs(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        raw_document = json.load(f)
    validated_document = [Document.model_validate(item) for item in raw_document]
    return validated_document