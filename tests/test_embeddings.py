"""Tests for real embeddings + the embedding cache + semantic transfer (offline, mock embedder)."""
from swm.variables.embeddings import EmbeddingCache
from swm.variables.embedding_registry import EmbeddingPriorRegistry
from swm.variables.prior_registry import PriorRegistry


def test_embedding_cache_roundtrip_and_fallback(tmp_path):
    path = str(tmp_path / "e.json")
    cache = EmbeddingCache.load(path)
    cache.precompute(["a", "b"], lambda batch: [[1.0, 0.0] for _ in batch])
    cache.save()
    reloaded = EmbeddingCache.load(path)
    assert reloaded.vecs["a"] == [1.0, 0.0]
    fn = reloaded.embed_fn()                                  # no live fn
    assert fn("a") == [1.0, 0.0] and fn("unknown") is None    # cached hit; miss -> None (never mixes spaces)


def test_semantic_transfer_via_embeddings():
    # a mock embedder where a synonym query points almost the same direction as the stored key
    vecs = {"inflation rate hike": [1.0, 0.0, 0.0],
            "price growth monetary tightening": [0.98, 0.05, 0.0],
            "unrelated thing": [0.0, 0.0, 1.0]}
    embed = lambda t: vecs.get(t)
    base = PriorRegistry()
    base.update("inflation", "rate hike", mean=1.5, sd=0.2, n=500)
    reg = EmbeddingPriorRegistry(base, embed_fn=embed, threshold=0.8).build_index()
    got = reg.get("price growth", "monetary tightening")     # embeds "price growth monetary tightening"
    assert got is not None and got.mean == 1.5 and "transfer" in got.source
    assert reg.get("unrelated", "thing") is None             # far away -> no spurious transfer
