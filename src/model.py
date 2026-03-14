import numpy as np


def initialize_embeddings(vocab_size: int, embedding_dim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Create input and output embedding matrices with small random values."""
    input_embeddings = np.random.uniform(-0.01, 0.01, (vocab_size, embedding_dim))
    output_embeddings = np.random.uniform(-0.01, 0.01, (vocab_size, embedding_dim))
    return input_embeddings, output_embeddings


def sample_negative_words(vocab_size: int, positive_word_id: int, num_negative: int = 3) -> list[int]:
    """Sample random negative word ids that are different from the positive context word."""
    negatives: list[int] = []

    while len(negatives) < num_negative:
        random_id = np.random.randint(0, vocab_size)
        if random_id != positive_word_id:
            negatives.append(random_id)

    return negatives


def sigmoid(x: float) -> float:
    """Compute the sigmoid function for a scalar value."""
    return 1.0 / (1.0 + np.exp(-x))


def compute_score(input_vector: np.ndarray, output_vector: np.ndarray) -> float:
    """Compute dot-product similarity score between two word vectors."""
    return float(np.dot(input_vector, output_vector))


def compute_loss(positive_score: float, negative_scores: list[float]) -> float:
    """Compute skip-gram loss with negative sampling for one training example."""
    positive_loss = -np.log(sigmoid(positive_score) + 1e-10)

    negative_loss = 0.0
    for score in negative_scores:
        negative_loss += -np.log(sigmoid(-score) + 1e-10)

    return float(positive_loss + negative_loss)