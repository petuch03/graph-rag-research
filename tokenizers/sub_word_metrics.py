from types import SimpleNamespace

import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical
from torch import Tensor


def threshold_noise_metric(attention: Tensor, threshold: float = 0.01):
    """
    Computes the noise metric for attention weights by identifying low-attention relationships.
    The function calculates for each token, across all heads and layers, the percentage of relationships
    (both as a source and as a target) that fall below a specified attention threshold, considering
    these low-attention relationships as noise.

    Parameters:
    - attention (tuple): A tuple of attention weights with the shape
      [num_layers, batch_size, num_heads, seq_len, seq_len], where seq_len is the sequence length.
    - threshold (float, optional): A threshold value to identify low-attention weights.
      Defaults to 0.01.

    Returns:
    - SimpleNamespace: An object with two attributes:
        - source_noise_percentage: A tensor of shape [seq_len] representing the average percentage of
          low-attention relationships for each token as a source across all heads, layers and batches.
        - target_noise_percentage: A tensor of shape [seq_len] representing the average percentage of
          low-attention relationships for each token as a target across all heads, layers and batches.
    """
    num_layers, batch_size, num_heads, seq_len, _ = attention.size()

    # Initialize tensors to accumulate scores
    source_low_attention_counts = torch.zeros(seq_len)
    target_low_attention_counts = torch.zeros(seq_len)

    for layer in range(num_layers):
        for batch in range(batch_size):
            for head in range(num_heads):
                for source_token in range(seq_len):  # Iterate over source tokens
                    for target_token in range(seq_len):  # Iterate over target tokens
                        # Count if attention value is below threshold, indicating noise
                        if attention[layer, batch, head, source_token, target_token] < threshold:
                            source_low_attention_counts[source_token] += 1
                            target_low_attention_counts[target_token] += 1

    # Calculate the percentage of noise relationships for source and target
    source_noise_percentage = (source_low_attention_counts / (seq_len * num_heads * num_layers))
    target_noise_percentage = (target_low_attention_counts / (seq_len * num_heads * num_layers))

    return SimpleNamespace(source_noise_percentage_per_token=source_noise_percentage,
                           target_noise_percentage_per_token=target_noise_percentage,
                           source_noise_percentage_over_tokens=source_noise_percentage.mean(),
                           target_noise_percentage_over_tokens=target_noise_percentage.mean())


def entropy_based_noise_metric(attention: Tensor) -> float:
    """
    Calculates the entropy of attention weights for each head and layer as a measure of noise.
    Higher entropy values indicate more uniform attention distributions (potentially more noise),
    while lower values indicate more focused attention.

    Parameters:
    - attention (tuple): A tensor of attention weights with the shape
      [num_layers, num_heads, seq_len, seq_len].

    Returns:
    - float: The average entropy across all heads and layers, providing a scalar value that
      represents the overall noise level of the attention mechanism based on entropy.
    """
    entropies = []
    for layer in attention:
        for head in layer:
            # Apply softmax to attention weights if not already done. This standardizes the distribution.
            attn_weights = softmax(head, dim=-1)
            dist = Categorical(probs=attn_weights)
            entropy = dist.entropy()  # Calculate entropy for each token
            entropies.append(entropy.mean())  # Average entropy across all tokens in the head
    # Average entropy across all heads and layers
    result = torch.stack(entropies).mean().item()

    return result


def show_all_metrics(attention: Tensor, threshold: float = 0.01):
    """
    Utility function to compute and print both threshold-based and entropy-based noise metrics for a given
    set of attention weights. This function serves as a convenience wrapper around `threshold_noise_metric`
    and `entropy_based_noise_metric` functions.

    Parameters:
    - attention (Tensor): A tensor of attention weights with the shape
      [num_layers, num_heads, seq_len, seq_len].
    - threshold (float, optional): A threshold value to identify low-attention weights for the
      threshold-based noise metric. Defaults to 0.01.

    Returns:
    - None: This function directly prints the results of the noise metrics computation but does not return any value.
    """
    # Call the threshold noise metric function and print its results
    threshold_results = threshold_noise_metric(attention, threshold)
    print(f"Threshold-based Noise Metric - Source: {threshold_results.source_noise_percentage_per_token}, "
          f"Source mean: {threshold_results.source_noise_percentage_over_tokens}\n"
          f"Target: {threshold_results.target_noise_percentage_per_token}"
          f"Target mean: {threshold_results.target_noise_percentage_over_tokens}")

    # Call the entropy based noise metric function and print its result
    entropy_result = entropy_based_noise_metric(attention)
    print(f"Entropy-based Noise Metric: {entropy_result}")
