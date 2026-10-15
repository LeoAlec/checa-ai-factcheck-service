import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_retrieval_benchmark as brb  # noqa: E402


def test_clean_claim_strips_noise():
    assert brb.clean_claim("  É #FAKE que vacina causa magnetismo #boato?  ") == "vacina causa magnetismo"
    assert brb.clean_claim("É falso que o TSE alterou votos") == "o TSE alterou votos"
    assert brb.clean_claim("'Governo vai taxar o Pix'") == "Governo vai taxar o Pix"


def test_benchmark_composition(tmp_path):
    # Cria arquivos temporários mockados
    silver_file = tmp_path / "silver.jsonl"
    recogna_file = tmp_path / "recogna.xlsx"
    faketrue_file = tmp_path / "faketrue.csv"

    import pandas as pd

    silver_urls = [f"https://checagem.example/{i}" for i in range(10)]
    with open(silver_file, "w", encoding="utf-8") as f:
        for u in silver_urls:
            f.write(json.dumps({"url": u, "titulo": f"Titulo {u}"}) + "\n")

    # Mock FakeRecogna
    recogna_rows = []
    # 5 fake in silver
    for i in range(5):
        recogna_rows.append({
            "Titulo": f"Boato sobre evento {i} #boato",
            "URL": silver_urls[i],
            "Classe": 0,
            "Categoria": "politica",
        })
    # 5 true not in silver
    for i in range(5):
        recogna_rows.append({
            "Titulo": f"Noticia veridica sobre economia {i}",
            "URL": f"https://noticiareal.example/{i}",
            "Classe": 1,
            "Categoria": "economia",
        })
    pd.DataFrame(recogna_rows).to_excel(recogna_file, index=False)

    # Mock FakeTrueBr
    faketrue_rows = []
    for i in range(5):
        faketrue_rows.append({
            "title_fake": f"Boato viral de whatsapp {i}",
            "link_f": f"https://boato-fora.example/{i}",
        })
    pd.DataFrame(faketrue_rows).to_csv(faketrue_file, index=False)

    cases, manifest = brb.build_benchmark(
        silver_path=silver_file,
        recogna_path=recogna_file,
        faketrue_path=faketrue_file,
        orig_eval_path=None,
        seed=42,
    )

    assert len(cases) > 0
    assert manifest["total_cases"] == len(cases)
    matched = [c for c in cases if c["expected_status"] == "matched"]
    abstained = [c for c in cases if c["expected_status"] == "abstained"]

    assert all(c["expected_url"] is not None for c in matched)
    assert all(c["expected_url"] is None for c in abstained)
