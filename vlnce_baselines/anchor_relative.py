import math


def anchor_relative_advantages(rewards, epsilon=1e-8):
    """Normalize each repair return by its delta from sample-zero E0."""
    advantages = [[0.0] * len(rewards[0]) for _ in rewards]
    for env_index, anchor_reward in enumerate(rewards[0]):
        if anchor_reward is None:
            continue
        deltas = []
        sample_indices = []
        for sample_index in range(1, len(rewards)):
            reward = rewards[sample_index][env_index]
            if reward is not None:
                deltas.append(reward - anchor_reward)
                sample_indices.append(sample_index)
        if not deltas:
            continue
        rms = math.sqrt(sum(delta * delta for delta in deltas) / len(deltas))
        denominator = rms + epsilon
        for sample_index, delta in zip(sample_indices, deltas):
            advantages[sample_index][env_index] = delta / denominator
    return advantages
