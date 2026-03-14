import re
from collections import Counter


def read_text(file_path: str) -> str:
    """Read raw text from a file and return it as a string."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def tokenize(text: str) -> list[str]:
    """Convert raw text into a cleaned list of lowercase tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


def build_vocab(tokens: list[str], min_count: int = 1) -> tuple[dict[str, int], dict[int, str]]:
    """Build word-to-id and id-to-word vocabularies from tokenized text."""
    counts = Counter(tokens)
    vocab_words = [word for word, count in counts.items() if count >= min_count]

    word_to_id = {word: idx for idx, word in enumerate(vocab_words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    return word_to_id, id_to_word


def encode_tokens(tokens: list[str], word_to_id: dict[str, int]) -> list[int]:
    """Convert tokens into integer ids using the vocabulary mapping."""
    return [word_to_id[word] for word in tokens if word in word_to_id]


def generate_skipgram_pairs(token_ids: list[int], window_size: int = 2) -> list[tuple[int, int]]:
    """Generate (center, context) training pairs for the skip-gram model."""
    pairs: list[tuple[int, int]] = []

    for center_idx in range(len(token_ids)):
        center_word = token_ids[center_idx]

        left = max(0, center_idx - window_size)
        right = min(len(token_ids), center_idx + window_size + 1)

        for context_idx in range(left, right):
            if context_idx == center_idx:
                continue
            context_word = token_ids[context_idx]
            pairs.append((center_word, context_word))

    return pairs