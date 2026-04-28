from pathlib import Path


def infer_category(path: Path) -> str:
    return path.parent.name


def build_chunk_id(doc_name: str, chunk_index: int) -> str:
    return f"{doc_name}#chunk-{chunk_index}"


def build_image_chunk_id(doc_name: str, image_index: int) -> str:
    return f"{doc_name}#image-{image_index}"
