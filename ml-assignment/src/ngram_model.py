import random
import re
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Optional


class TrigramModel:
    """
    A simple trigram (n=3) language model.

    - Uses (w1, w2) -> {w3: count} as the core data structure.
    - Handles:
        * lowercasing
        * basic punctuation stripping
        * unknown word mapping via <unk>
        * start and end padding tokens
    - Generates text by sampling probabilistically from the trigram
      distribution instead of picking the argmax word.
    """

    def __init__(self, unk_threshold: int = 1) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        unk_threshold : int
            Words that appear <= unk_threshold times in the training
            corpus are mapped to the <unk> token.
        """
        # (w1, w2) -> {w3: count}
        self.trigram_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # (w1, w2) -> total count of all w3 for this context
        self.bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        # Vocabulary of known tokens (after UNK mapping)
        self.vocab = set()

        # Special tokens
        self.start_tokens = ("<s1>", "<s2>")
        self.end_token = "</s>"
        self.unk_token = "<unk>"

        self.unk_threshold = unk_threshold
        self.trained = False

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, text: str) -> List[str]:
        """
        Basic text cleaning: lowercase + remove punctuation.

        We keep it intentionally simple:
        - lowercase everything
        - replace non-word / non-space chars with a space
        - split on whitespace
        """
        text = text.lower()
        # Replace punctuation with spaces so words don't get glued together
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return tokens

    def _build_vocab(self, tokens: List[str]) -> List[str]:
        """
        Build vocabulary and map rare words to <unk>.

        Returns the token sequence with rare words replaced by <unk>.
        """
        counts = Counter(tokens)

        # Keep tokens that appear more than unk_threshold
        self.vocab = {w for w, c in counts.items() if c > self.unk_threshold}

        # Always make sure <unk> is in the vocab
        self.vocab.add(self.unk_token)

        # Map rare tokens to <unk>
        mapped_tokens = [w if w in self.vocab else self.unk_token for w in tokens]
        return mapped_tokens

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, text: str) -> None:
        """
        Train the trigram model on the given raw text.

        Steps:
        1. Preprocess text into tokens.
        2. Build vocab + map rare words to <unk>.
        3. Add start and end padding.
        4. Count trigram and bigram frequencies.
        """
        tokens = self._preprocess(text)

        if not tokens:
            # Empty text edge case
            self.trained = False
            return

        # Build vocab and replace rare tokens with <unk>
        tokens = self._build_vocab(tokens)

        # For simplicity we treat the whole corpus as one long sequence,
        # but pad it with start and end tokens.
        sequence: List[str] = [*self.start_tokens, *tokens, self.end_token]

        # Count trigrams and bigrams
        for i in range(len(sequence) - 2):
            w1, w2, w3 = sequence[i], sequence[i + 1], sequence[i + 2]
            context = (w1, w2)
            self.trigram_counts[context][w3] += 1
            self.bigram_counts[context] += 1

        self.trained = True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample_next(self, context: Tuple[str, str]) -> Optional[str]:
        """
        Sample the next word given a (w1, w2) context using the
        trigram counts as a probability distribution.

        Returns
        -------
        str or None
            The next token, or None if the context was never seen.
        """
        next_dict = self.trigram_counts.get(context)
        if not next_dict:
            # Unseen context -> we currently give up.
            # (Could be improved with backoff / fallback.)
            return None

        total = self.bigram_counts[context]
        if total == 0:
            return None

        # Draw a random number in [0, 1) and walk the CDF
        r = random.random()
        cumulative = 0.0

        # dict preserves insertion order in modern Python, which is fine here
        for word, count in next_dict.items():
            prob = count / total
            cumulative += prob
            if r <= cumulative:
                return word

        # Numerical edge-case fallback
        # (if we got here, just return the last word)
        return word

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate(self, max_length: int = 50) -> str:
        """
        Generate a piece of text using the trained trigram model.

        Parameters
        ----------
        max_length : int
            Maximum number of tokens to generate (not counting start tokens).

        Returns
        -------
        str
            Generated text as a single string.
        """
        if not self.trained:
            return ""

        w1, w2 = self.start_tokens
        generated_tokens: List[str] = []

        for _ in range(max_length):
            next_word = self._sample_next((w1, w2))
            if not next_word:
                break

            if next_word == self.end_token:
                break

            generated_tokens.append(next_word)

            # Move the window forward
            w1, w2 = w2, next_word

        return " ".join(generated_tokens)
