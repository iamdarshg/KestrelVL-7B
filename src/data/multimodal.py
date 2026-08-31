"""Mixture weights and held-out set policy for multimodal data."""

MIXTURE_WEIGHTS = {
    "codervision": 0.30,
    "nemotron_image_selected": 0.20,
    "websight": 0.15,
    "ocr_document": 0.10,
    "handwriting_coderink": 0.10,
    "gui_grounding": 0.10,
    "general_vision": 0.05,
}
HELD_OUT_DATASETS = {"screenspot-pro", "mmmu-pro", "osworld", "gpqa", "terminal-bench", "swe-bench"}

