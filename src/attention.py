import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# mRNA Nucleotide Vocabulary mapping to embedding indices
VOCAB = ['A', 'C', 'G', 'U']
EMBED_DIM = 8

# Toy Learnable Embedding Lookup Table (one vector per nucleotide)
EMBEDDING_TABLE = {base: np.random.randn(EMBED_DIM) for base in VOCAB}

def encode_sequence(sequence: str) -> np.ndarray:
    """
    Encode an mRNA nucleotide sequence into an embedding matrix (FR-2.1, FR-2.2).

    Args:
        sequence (str): String containing only characters A, C, G, U (length 5-50).
    
    Returns:
        np.ndarray: Matrix of shape (seq_len, EMBED_DIM).
    """
    if not isinstance(sequence, str):
        raise TypeError("Sequence input must be a string.")

    if not (5 <= len(sequence) <= 50):
        raise ValueError(f"Sequence length must be between 5 and 50 characters. Got {len(sequence)}.")

    invalid_chars = set(sequence) - set(VOCAB)
    if invalid_chars:
        raise ValueError(f"Invalid nucleotide character(s) detected: {invalid_chars}. Allowed: {VOCAB}")

    return np.stack([EMBEDDING_TABLE[base] for base in sequence])

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax along the last axis."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention(X: np.ndarray, d_k: int=EMBED_DIM):
    """
    Compute Scaled Dot-Product Self-Attention (FR-2.3, FR-2.4).

    Formula: Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k)) @ V

    Args:
        X (np.ndarray): Input sequence embeddings of shape (seq_len, embed_dim).
        d_k (int): Dimensionality of Query/Key vectors.
    
    Returns:
        tuple: (attention_weights, output)
            - attention_weights: Shape (seq_len, seq_len)
            - output: Shape (seq_len, d_k)
    """
    seq_len, embed_dim = X.shape

    # Intialize projection weights for Q, K, V
    W_q = np.random.randn(embed_dim, d_k) * 0.1
    W_k = np.random.randn(embed_dim, d_k) * 0.1
    W_v = np.random.randn(embed_dim, d_k) * 0.1

    # Project input embeddings to Q, K, V
    Q = X @ W_q     # (seq_len, d_k)
    K = X @ W_k     # (seq_len, d_k)
    V = X @ W_v     # (seq_len, d_k)

    # Compute scaled dot-product attention scores
    scores = (Q @ K.T) / np.sqrt(d_k)

    # Apply softmax to obtain probability distributions (row sum to 1.0)
    attention_weights = softmax(scores)

    # Weighted sum of Value vectors
    output = attention_weights @ V      # (seq_len, d_k)

    return attention_weights, output

def plot_attention(attention_weights: np.ndarray, sequence: str, save_path: str):
    """Render and save attention weights as a heatmap (FR-2.5)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention_weights, cmap="viridis")

    # Label axes with nucleotide characters
    ax.set_xticks(range(len(sequence)))
    ax.set_yticks(range(len(sequence)))
    ax.set_xticklabels(list(sequence))
    ax.set_yticklabels(list(sequence))

    ax.set_xlabel("Attending to Position")
    ax.set_ylabel("Attending from Position")
    ax.set_title("mRNA Sequence Self-Attention Heatmap")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", rotation=270, labelpad=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"Attention heatmap saved successfully to '{save_path}'")

if __name__ == "__main__":
    # Test Sequence
    sample_sequence = "ACGUACGUACGGCUA"
    print(f"Testing self-attention engine with sequence: {sample_sequence} (Len: {len(sample_sequence)})")

    # Step 1: Embed Sequence
    X = encode_sequence(sample_sequence)
    print(f"Embedded Matrix shape: {X.shape}")

    # Step 2: Compute self-attention
    weights, attn_output = self_attention(X)
    print(f"Attention weights matrix shape: {weights.shape}")

    # Step 3: Validate row sum requirement (FR-2.4)
    row_sums = weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Softmax sanity check failed! Row sums! {row_sums}"
    print("Sanity Check Passed: All rows sum strictly to 1.0 within floating-point tolerance.")

    # Step 4: Plot heatmap (FR-2.5)
    heatmap_path = "outputs/figures/attention_heatmap.png"
    plot_attention(weights, sample_sequence, heatmap_path)