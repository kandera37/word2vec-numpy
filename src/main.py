from data_utils import read_text, tokenize, build_vocab, encode_tokens, generate_skipgram_pairs
from model import (
    initialize_embeddings,
    sample_negative_words,
    compute_score,
    sigmoid,
    compute_loss,
    train_one_step,
    train_epochs,
)

def main() -> None:
    """Run a small skip-gram word2vec demo with negative sampling."""
    text = read_text("data/sample.txt")

    tokens = tokenize(text)
    word_to_id, id_to_word = build_vocab(tokens)
    token_ids = encode_tokens(tokens, word_to_id)
    pairs = generate_skipgram_pairs(token_ids, window_size=2)

    print("Tokens:", tokens)
    print("Vocabulary:", word_to_id)
    print("First 10 pairs:")
    for center_id, context_id in pairs[:10]:
        print(f"{id_to_word[center_id]} -> {id_to_word[context_id]}")

    vocab_size = len(word_to_id)
    input_embeddings, output_embeddings = initialize_embeddings(vocab_size, embedding_dim=8)

    print("Input embeddings shape:", input_embeddings.shape)
    print("Output embeddings shape:", output_embeddings.shape)

    center_id, context_id = pairs[0]
    negative_ids = sample_negative_words(vocab_size, context_id, num_negative=3)

    print("Example pair:", id_to_word[center_id], "->", id_to_word[context_id])
    print("Negative samples:", [id_to_word[idx] for idx in negative_ids])

    center_vector = input_embeddings[center_id]
    positive_vector = output_embeddings[context_id]
    negative_scores: list[float] = []

    positive_score = compute_score(center_vector, positive_vector)
    positive_probability = sigmoid(positive_score)

    print("Positive score:", positive_score)
    print("Positive probability:", positive_probability)

    print("Negative scores and probabilities:")
    for negative_id in negative_ids:
        negative_vector = output_embeddings[negative_id]
        negative_score = compute_score(center_vector, negative_vector)
        negative_scores.append(negative_score)
        negative_probability = sigmoid(negative_score)

        print(
            f"{id_to_word[center_id]} -> {id_to_word[negative_id]} | "
            f"score={negative_score:.6f}, prob={negative_probability:.6f}"
        )

    loss = compute_loss(positive_score, negative_scores)
    print("Loss:", loss)

    updated_loss = train_one_step(
        center_id=center_id,
        context_id=context_id,
        negative_ids=negative_ids,
        input_embeddings=input_embeddings,
        output_embeddings=output_embeddings,
        learning_rate=0.05,
    )

    print("Loss before update:", loss)

    center_vector_after = input_embeddings[center_id]
    positive_vector_after = output_embeddings[context_id]
    negative_scores_after: list[float] = []

    positive_score_after = compute_score(center_vector_after, positive_vector_after)
    positive_probability_after = sigmoid(positive_score_after)

    for negative_id in negative_ids:
        negative_vector_after = output_embeddings[negative_id]
        negative_score_after = compute_score(center_vector_after, negative_vector_after)
        negative_scores_after.append(negative_score_after)

    loss_after = compute_loss(positive_score_after, negative_scores_after)

    print("Loss returned by training step:", updated_loss)
    print("Loss after update:", loss_after)

    epoch_losses = train_epochs(
        pairs=pairs,
        input_embeddings=input_embeddings,
        output_embeddings=output_embeddings,
        vocab_size=vocab_size,
        num_negative=3,
        learning_rate=0.05,
        epochs=5,
    )

    print("Epoch losses:")
    for epoch_index, epoch_loss in enumerate(epoch_losses, start=1):
        print(f"Epoch {epoch_index}: {epoch_loss:.6f}")
        
if __name__ == "__main__":
    main()