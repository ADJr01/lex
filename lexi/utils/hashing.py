import hashlib


def chunk_id(file_path: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(f"{file_path}:{chunk_index}:{text}".encode())
    return h.hexdigest()
