import pytest
from pydantic import ValidationError

from llm_summarizer.models import ChatCompletion


def test_empty_choices_is_rejected():
    with pytest.raises(ValidationError):
        ChatCompletion.model_validate({"choices": []})
    