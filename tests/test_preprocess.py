from triagedesk.ml.preprocess import normalize_text


def test_normalize_preserves_negation_and_masks_sensitive_patterns():
    value = normalize_text(
        "  I do NOT recognize 4111 1111 1111 1111; mail me at User@example.com  "
    )
    assert "NOT" in value
    assert "<NUMBER>" in value
    assert "<EMAIL>" in value
    assert "  " not in value
