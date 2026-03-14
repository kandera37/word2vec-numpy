from data_utils import read_text, tokenize, build_vocab, encode_tokens, generate_skipgram_pairs
from model import initialize_embeddings, sample_negative_words, compute_score, sigmoid, compute_loss


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


if __name__ == "__main__":
    main()