def test_patch_qdrant_local_compat_strips_extra_meta_fields():
    """Executa em processo limpo, pois testes legados simulam qdrant_client."""
    import subprocess
    import sys

    program = r'''
from src.rag.qdrant import patch_qdrant_local_compat
patch_qdrant_local_compat()
import qdrant_client.local.qdrant_local as ql
config = ql.rest_models.CreateCollection(
    vectors={"size": 1536, "distance": "Cosine", "memory": None},
    payload=None,
    metadata=None,
)
assert config.vectors.size == 1536
assert str(config.vectors.distance) in {"Distance.COSINE", "Cosine"}
'''
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
