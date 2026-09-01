import math

import numpy as np


def recoverable_outcome(positions, goal_distances, success_distance):
    """Return success, best prefix SPL, and closest goal distance."""
    positions = np.asarray(positions, dtype=np.float64)
    goal_distances = np.asarray(goal_distances, dtype=np.float64)
    if positions.ndim != 2 or len(positions) != len(goal_distances):
        raise ValueError('positions and goal_distances must align')

    segment_lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    prefix_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    successful = goal_distances <= success_distance
    min_distance = float(goal_distances.min())
    if not successful.any():
        return False, 0.0, min_distance

    shortest_path_length = float(goal_distances[0])
    best_path_length = float(prefix_lengths[successful].min())
    best_spl = shortest_path_length / max(
        shortest_path_length, best_path_length, 1e-8
    )
    return True, best_spl, min_distance


def _outcome_key(outcome):
    success, best_spl, min_distance = outcome
    if success:
        return 1, float(best_spl), 0.0
    return 0, 0.0, -float(min_distance)


def setwise_group_advantages(outcomes, epsilon=1e-8):
    """Rank rollout outcomes per environment and standardize their ranks."""
    advantages = [[0.0] * len(outcomes[0]) for _ in outcomes]
    active_groups = [False] * len(outcomes[0])

    for env_index in range(len(outcomes[0])):
        ranked = [
            (sample_index, _outcome_key(sample[env_index]))
            for sample_index, sample in enumerate(outcomes)
            if sample[env_index] is not None
        ]
        ranked.sort(key=lambda item: item[1])
        if len(ranked) < 2:
            continue

        ranks = {}
        start = 0
        while start < len(ranked):
            end = start + 1
            while end < len(ranked) and ranked[end][1] == ranked[start][1]:
                end += 1
            average_rank = 0.5 * (start + end - 1)
            for index in range(start, end):
                ranks[ranked[index][0]] = average_rank
            start = end

        mean_rank = sum(ranks.values()) / len(ranks)
        variance = sum(
            (rank - mean_rank) ** 2 for rank in ranks.values()
        ) / len(ranks)
        std_rank = math.sqrt(variance)
        if std_rank <= epsilon:
            continue

        active_groups[env_index] = True
        for sample_index, rank in ranks.items():
            advantages[sample_index][env_index] = (
                rank - mean_rank
            ) / (std_rank + epsilon)

    return advantages, active_groups
