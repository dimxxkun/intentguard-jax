"""Export only verified inference artifacts, without datasets or local run history."""

import argparse
import zipfile
from pathlib import Path

from intentguard.evaluation import verify_bundle
from intentguard.io import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle, output = Path(args.bundle), Path(args.output)
    project = Path(__file__).resolve().parents[1]
    manifest = verify_bundle(bundle)
    if output.exists():
        raise ValueError("Export path already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("bundle.json", *manifest["files"]):
            archive.write(bundle / name, f"intentguard/{name}")
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "docs/model-card.md"):
            archive.write(project / name, name)
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(f"{output.name}: {output.stat().st_size:,} bytes; SHA-256 {digest}")


if __name__ == "__main__":
    main()
