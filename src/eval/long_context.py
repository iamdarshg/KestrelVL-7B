def niah_score(prediction: str, needle: str) -> float:
    return float(needle in prediction)
