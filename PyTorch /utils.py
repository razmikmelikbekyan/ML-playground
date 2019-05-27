import torch


def check_relative_difference(a: torch.tensor, b: torch.tensor, threshold: float) -> bool:
    """Returns True if (|a - b| / (|a| + |b|)) > threshold else False."""
    numerator = torch.abs(a - b)
    denominator = torch.abs(a) + torch.abs(b)
    result = numerator / denominator
    result[torch.isnan(result)] = 0
    return bool(torch.any(result > threshold))
