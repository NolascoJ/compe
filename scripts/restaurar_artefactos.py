"""Restaura artefactos comprimidos y verifica sus hashes, sin entrenar modelos."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def restore(check_only: bool = False) -> None:
    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    for entry in manifest["files"]:
        target = ROOT / entry["path"]
        compressed = ROOT / entry["compressed_path"]
        if digest(compressed) != entry["compressed_sha256"]:
            raise ValueError(f"Hash comprimido incorrecto: {entry['compressed_path']}")
        if target.exists() and digest(target) != entry["sha256"]:
            raise ValueError(f"Existe un archivo diferente; no se sobrescribe: {entry['path']}")
        h = hashlib.sha256()
        size = 0
        temporary = None
        try:
            if not check_only and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = tempfile.NamedTemporaryFile(dir=target.parent, delete=False)
            with gzip.open(compressed, "rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    h.update(block)
                    size += len(block)
                    if temporary is not None:
                        temporary.write(block)
            if size != entry["size_bytes"] or h.hexdigest() != entry["sha256"]:
                raise ValueError(f"Contenido restaurado incorrecto: {entry['path']}")
            if temporary is not None:
                temporary.close()
                os.replace(temporary.name, target)
            print(f"{'Verificado' if check_only else 'Disponible'}: {entry['path']}")
        finally:
            if temporary is not None:
                temporary.close()
                Path(temporary.name).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verificar sin restaurar archivos.")
    restore(parser.parse_args().check)
