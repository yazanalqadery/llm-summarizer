import json

from llm_summarizer.models import AnalyzedDoc
from llm_summarizer.pipeline import save_results


def test_save_results_writes_analysis_fields(tmp_path):
    doc = AnalyzedDoc(id=1, title="A", summary="s", keywords=["a", "b"], sentiment="neutral")
    out = tmp_path / "out.json"
    save_results([doc], [], str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summaries"][0]["keywords"] == ["a", "b"]   
    assert data["failures"] == []                    
    assert list(data.keys()) == ["summaries", "failures"]                  