import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "local_llm_verdict_colab.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_local_llm_notebook_handles_batch_encoding_from_chat_template():
    """The chat template returns a mapping, not a tensor with a ``shape`` field."""
    source = _notebook_source()

    assert "return_dict=True" in source
    assert "input_length = encoded['input_ids'].shape[-1]" in source
    assert "model.generate(**encoded" in source
    assert "encoded.shape" not in source


def test_local_llm_notebook_accepts_canonical_and_flattened_benchmark_rows():
    source = _notebook_source()

    assert "evidence = item.get('evidence', {})" in source
    assert "item.get('evidence_title', evidence.get('titulo', ''))" in source
    assert "item.get('evidence_text', evidence.get('texto', ''))" in source
