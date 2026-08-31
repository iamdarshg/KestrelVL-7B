def pass_rate(results: list[bool]) -> float:
    return sum(results) / len(results) if results else 0.0
