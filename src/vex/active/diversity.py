"""Diversity: embed first pages, cluster them, and pick across clusters.

Picking only the most uncertain documents tends to pick many near-duplicates
(the same layout failing the same way). Clustering page images and taking
documents cluster by cluster spreads the batch over layouts.

Embedders (`active.diversity.embedder`):
- `onnx`: ResNet50 exported to ONNX (`Qdrant/resnet50-onnx`, 94 MB, downloaded
  once to the Hugging Face cache), run with onnxruntime. Optional group:
  `uv sync --extra embed`. fastembed, which wraps the same model, cannot be used:
  its image support needs pillow>=10.3 and Kili 2.142.1 pins pillow<10.1;
- `thumbnail`: 32×32 grayscale pixels of page 1. No extra dependency, and
  enough to separate layouts that look different;
- `auto`: `onnx` if its dependencies are installed, else `thumbnail`.
"""

import io
from pathlib import Path

import numpy as np
import pymupdf
from loguru import logger
from PIL import Image
from sklearn.cluster import KMeans


ONNX_REPO = "Qdrant/resnet50-onnx"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _first_page(path: Path, dpi: int = 60) -> Image.Image:
    # Only page 1 is embedded, by design: render it directly rather than through
    # load_pages(max_pages=1), which would warn about every multi-page PDF.
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as pdf:
            return Image.open(io.BytesIO(pdf[0].get_pixmap(dpi=dpi).tobytes("png")))
    return Image.open(path)


def _l2(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms == 0, 1, norms)


def thumbnail_embeddings(paths: list[Path]) -> np.ndarray:
    """32×32 grayscale thumbnails of page 1, flattened and L2-normalized.

    Args:
        paths: Document files.

    Returns:
        Array of shape (len(paths), 1024).
    """
    rows = [np.asarray(_first_page(p).convert("L").resize((32, 32), Image.Resampling.BILINEAR), np.float32).ravel()
            for p in paths]
    return _l2(np.stack(rows) / 255.0)


def _preprocess(image: Image.Image) -> np.ndarray:
    # Mirrors the model's preprocessor_config.json: shortest edge 224/0.875 = 256
    # (bicubic), centre crop 224, rescale to [0, 1], ImageNet mean/std, CHW.
    image = image.convert("RGB")
    scale = 256 / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
    left, top = (image.width - 224) // 2, (image.height - 224) // 2
    pixels = np.asarray(image.crop((left, top, left + 224, top + 224)), np.float32) / 255.0
    return ((pixels - _MEAN) / _STD).transpose(2, 0, 1)


def onnx_embeddings(paths: list[Path]) -> np.ndarray:
    """ResNet50 (ONNX) embeddings of page 1, L2-normalized.

    Args:
        paths: Document files.

    Returns:
        Array of shape (len(paths), 2048).

    Raises:
        ImportError: If the `embed` extra is not installed.
    """
    import onnxruntime  # noqa: PLC0415  (optional dependency)
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    session = onnxruntime.InferenceSession(hf_hub_download(ONNX_REPO, "model.onnx"),
                                           providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rows = [session.run(None, {input_name: _preprocess(_first_page(p))[None]})[0].reshape(-1) for p in paths]
    return _l2(np.stack(rows).astype(np.float32))


def embed_documents(paths: list[Path], embedder: str) -> tuple[np.ndarray, str]:
    """Embed page 1 of each document.

    Args:
        paths: Document files.
        embedder: `auto`, `onnx` or `thumbnail`.

    Returns:
        The embeddings and the embedder actually used.
    """
    if embedder in ("auto", "onnx"):
        try:
            return onnx_embeddings(paths), "onnx"
        except ImportError:
            if embedder == "onnx":
                raise
            logger.info("embed extra not installed: using thumbnail embeddings (uv sync --extra embed for ONNX)")
    return thumbnail_embeddings(paths), "thumbnail"


def pick_diverse(doc_ids: list[str], embeddings: np.ndarray, scores: dict[str, float], n_clusters: int, k: int,
                 seed: int) -> list[tuple[str, int]]:
    """Cluster documents and pick the most uncertain ones, rotating over clusters.

    Clusters are visited in order of their most uncertain document; in each
    pass, every cluster gives its next most uncertain document, until `k`
    documents are picked.

    Args:
        doc_ids: Candidate documents (rows of `embeddings`).
        embeddings: One row per document.
        scores: Combined uncertainty per document.
        n_clusters: Requested number of clusters (capped by the number of documents).
        k: Number of documents to pick.
        seed: KMeans seed.

    Returns:
        `(doc_id, cluster)` pairs, in pick order.
    """
    if not doc_ids or k <= 0:
        return []
    n = min(n_clusters, len(doc_ids))
    labels = KMeans(n_clusters=n, n_init=10, random_state=seed).fit_predict(embeddings) if n > 1 else np.zeros(len(doc_ids), int)
    members: dict[int, list[str]] = {}
    for doc_id, label in zip(doc_ids, labels, strict=True):
        members.setdefault(int(label), []).append(doc_id)
    for ids in members.values():
        ids.sort(key=lambda d: (-scores[d], d))
    order = sorted(members, key=lambda c: (-scores[members[c][0]], c))
    picked: list[tuple[str, int]] = []
    while len(picked) < k and any(members.values()):
        for cluster in order:
            if members[cluster] and len(picked) < k:
                picked.append((members[cluster].pop(0), cluster))
    return picked
