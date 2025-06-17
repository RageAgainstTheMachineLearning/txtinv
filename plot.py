import torch


def display_batch(data, tokenizer):
    """Displays the target text.
    """
    batch_size = data.shape[0]

    print("=" * 30)
    print("Current batch display:")
    for idx in range(batch_size):
        # Convert tensor to list
        input_ids = data[idx].tolist()
        # Decode the input IDs to text
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"{idx}: {decoded_text}")
        print("-" * 30)
    print("=" * 30)


def display_reconstruction(data, tokenizer, embedding_matrix):
    """Displays the reconstructed text.
    We derive original tokens by finding the closest entry in the embedding matrix reversely.

    Args:
        data (torch.Tensor): The input tensor that has shape [batch_size, seq_len, embedding_dim].
        tokenizer (Tokenizer): The tokenizer used for decoding.
        embedding_matrix (torch.Tensor): The embedding matrix used for reconstruction.
    """

    batch_size = data.shape[0]
    seq_len = data.shape[1]
    embedding_dim = data.shape[2]

    print("=" * 30)
    print("Reconstructed text display:")
    for idx in range(batch_size):
        # Get the current sequence
        current_seq = data[idx].view(seq_len, embedding_dim)
        reconstructed_tokens = []

        for token in current_seq:
            # Find the closest entry in the embedding matrix
            distances = torch.norm(embedding_matrix - token, dim=1)
            closest_idx = torch.argmin(distances).item()
            reconstructed_tokens.append(closest_idx)

        # Decode the reconstructed tokens to text
        decoded_text = tokenizer.decode(
            reconstructed_tokens, skip_special_tokens=True)
        print(f"{idx}: {decoded_text}")
        print("-" * 30)
    print("=" * 30)
