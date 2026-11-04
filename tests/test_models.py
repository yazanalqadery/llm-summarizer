import pytest
from pydantic import ValidationError

from llm_summarizer.models import Analysis, ChatCompletion

GOOD = (
    '{"summary": "Remote work saves commuting time.", '
    '"keywords": ["remote", "commute"], "sentiment": "neutral"}'
)
MISSING = '{"summary": "x", "keywords": ["a"]}'
EXTRA = (
    '{"summary": "x", "keywords": ["a"], "sentiment": "neutral", "mood": "happy"}'
)


def test_empty_choices_is_rejected():
    with pytest.raises(ValidationError):
        ChatCompletion.model_validate({"choices": []})


def test_analysis_accepts_valid_json():
    a = Analysis.model_validate_json(GOOD)
    assert a.keywords == ["remote", "commute"]


def test_analysis_rejects_missing_field():
    with pytest.raises(ValidationError):
        Analysis.model_validate_json(MISSING)


def test_analysis_rejects_extra_field():
    with pytest.raises(ValidationError):
        Analysis.model_validate_json(EXTRA)