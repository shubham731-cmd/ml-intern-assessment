import os
from pathlib import Path

from ngram_model import TrigramModel


def load_corpus(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    """
    Simple CLI script to train a trigram model on
    'Alice's Adventures in Wonderland' and generate sample text.

    Expected file location:
        ml-assignment/data/alice_in_wonderland.txt
    """
    # Resolve: /path/to/ml-intern-assessment-main/ml-assignment/data/alice_in_wonderland.txt
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "alice_in_wonderland.txt"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find corpus file at {data_path}. "
            "Please download 'Alice's Adventures in Wonderland' from Project Gutenberg "
            "and save it as 'alice_in_wonderland.txt' in the 'ml-assignment/data' folder."
        )

    print(f"Loading corpus from: {data_path}")
    text = load_corpus(data_path)

    # Create and train the model
    model = TrigramModel(unk_threshold=1)
    print("Training trigram model...")
    model.fit(text)
    print("Training complete.\n")

    # Generate a few sample sentences / segments
    for i in range(3):
        print(f"=== Sample #{i + 1} ===")
        generated = model.generate(max_length=40)
        print(generated)
        print()


if __name__ == "__main__":
    main()
