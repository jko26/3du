"""VRAM → DA3 Nested-Giant chunk size."""

from cork3du.reconstruct import chunk_size_for_vram_gib


def test_16gb_uses_tiny_chunks():
    assert chunk_size_for_vram_gib(15.56) == (4, 2)


def test_a100_40gb():
    assert chunk_size_for_vram_gib(40.0) == (12, 6)


def test_h100():
    assert chunk_size_for_vram_gib(80.0) == (24, 12)
