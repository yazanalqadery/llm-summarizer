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

def test_analysis_rejects_unknown_sentiment():
    bad = GOOD.replace('"neutral"', '"mixed"')
    with pytest.raises(ValidationError):
        Analysis.model_validate_json(bad)


def test_analysis_rejects_wrong_case_sentiment():
    bad = GOOD.replace('"neutral"', '"Neutral"')
    with pytest.raises(ValidationError):
        Analysis.model_validate_json(bad)

def test_analysis_accepts_all_three_sentiments():
    for sentiment in ["positive", "neutral", "negative"]:
        text = GOOD.replace('"neutral"', f'"{sentiment}"')
        assert Analysis.model_validate_json(text).sentiment == sentiment

def make_keywords_text(count: int) -> str:
    words = ", ".join(f'"k{i}"' for i in range(count))
    return GOOD.replace('["remote", "commute"]', f"[{words}]")


def test_analysis_accepts_five_keywords():
    a = Analysis.model_validate_json(make_keywords_text(5))
    assert len(a.keywords) == 5


def test_analysis_rejects_six_keywords():
    with pytest.raises(ValidationError):
        Analysis.model_validate_json(make_keywords_text(6))