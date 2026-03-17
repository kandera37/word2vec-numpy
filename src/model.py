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


def train_one_step(
    center_id: int,
    context_id: int,
    negative_ids: list[int],
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    learning_rate: float = 0.05,
) -> float:
    """Perform one skip-gram training step with negative sampling."""
    center_vector = input_embeddings[center_id].copy()
    positive_vector = output_embeddings[context_id].copy()
    negative_vectors = output_embeddings[negative_ids].copy()

    positive_score = compute_score(center_vector, positive_vector)
    negative_scores = [compute_score(center_vector, neg_vector) for neg_vector in negative_vectors]

    loss = compute_loss(positive_score, negative_scores)

    positive_grad = sigmoid(positive_score) - 1.0
    negative_grads = [sigmoid(score) for score in negative_scores]

    center_grad = positive_grad * positive_vector
    for neg_grad, neg_vector in zip(negative_grads, negative_vectors):
        center_grad += neg_grad * neg_vector

    input_embeddings[center_id] -= learning_rate * center_grad
    output_embeddings[context_id] -= learning_rate * (positive_grad * center_vector)

    for i, negative_id in enumerate(negative_ids):
        output_embeddings[negative_id] -= learning_rate * (negative_grads[i] * center_vector)

    return loss


def train_epochs(
    pairs: list[tuple[int, int]],
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    vocab_size: int,
    num_negative: int = 3,
    learning_rate: float = 0.05,
    epochs: int = 5,
) -> list[float]:
    """Train skip-gram embeddings for several epochs and return average loss per epoch."""
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        total_loss = 0.0

        for center_id, context_id in pairs:
            negative_ids = sample_negative_words(vocab_size, context_id, num_negative)
            loss = train_one_step(
                center_id=center_id,
                context_id=context_id,
                negative_ids=negative_ids,
                input_embeddings=input_embeddings,
                output_embeddings=output_embeddings,
                learning_rate=learning_rate,
            )
            total_loss += loss

        average_loss = total_loss / len(pairs)
        epoch_losses.append(float(average_loss))

    return epoch_losses