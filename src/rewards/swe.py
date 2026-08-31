def weighted_swe_reward(hidden_test_pass: float, compile_pass: float, localization: float, patch_valid: float, unnecessary_change: float, efficiency: float) -> float:
    return 0.70 * hidden_test_pass + 0.10 * compile_pass + 0.05 * localization + 0.05 * patch_valid + 0.05 * (1 - unnecessary_change) + 0.05 * efficiency

