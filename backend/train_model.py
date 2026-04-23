from __future__ import annotations

from pathlib import Path

from backend.ml_service import ResumeMLService, write_metadata_file


def main() -> None:
    service = ResumeMLService()
    model_path = service.train_from_source()
    metadata_path = Path(model_path).with_suffix(".json")
    write_metadata_file(metadata_path, service.export_metadata())
    print(f"Model written to {model_path}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
