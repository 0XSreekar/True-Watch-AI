"""Semantic footage search: SigLIP embeddings over sealed clips' keyframes.

embed.py    loads the pretrained SigLIP model once and turns images/text into
            L2-normalised vectors in a shared embedding space.
index.py    a small local vector index (numpy brute force) persisted to disk.
backfill.py batch- and incrementally-indexes clips found under a directory.
query.py    natural-language query -> ranked clips, in the exact shape the
            FootageSearch component already renders.

Everything here is local: the model is downloaded once (by whoever provisions
the box) and cached under the standard Hugging Face cache directory; no
network call happens at query time or index time after that first download.
"""
