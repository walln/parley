def test_import():
    import parley
    import parley.infer.generate
    import parley.infer.transport
    assert parley.__version__ if hasattr(parley, "__version__") else True


def test_constants():
    from parley.rows import SAMPLES_PER_FRAME, FRAME_MS
    assert SAMPLES_PER_FRAME == 1280
    assert FRAME_MS == 80