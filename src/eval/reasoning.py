def exact_match(prediction: str, answer: str) -> float:
    return float(prediction.strip() == answer.strip())
