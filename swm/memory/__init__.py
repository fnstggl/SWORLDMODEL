"""Episodic memory + reflection — the situation-conditioned recall layer (audit C.4).

See `swm/memory/memory.py` for the memory stream (recency × importance × relevance retrieval) and
generative reflection, `swm/memory/embeddings.py` for the dependency-free embedder, and
`swm/memory/retrieval_response.py` for the retrieval-augmented response_fn. Validated in EXP-074.
"""
from swm.memory.embeddings import TextEmbedder, cosine, hashing_embed, tokenize
from swm.memory.memory import Episode, EpisodicStore, LeakageError, MemoryStream
from swm.memory.retrieval_response import memory_signal, retrieval_augmented_response_fn

__all__ = ["TextEmbedder", "cosine", "hashing_embed", "tokenize", "Episode", "EpisodicStore",
           "LeakageError", "MemoryStream", "memory_signal", "retrieval_augmented_response_fn"]
