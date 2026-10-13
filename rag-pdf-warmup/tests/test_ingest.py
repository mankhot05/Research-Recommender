"""Chunking tests. Pure functions only: no PDF I/O, no OpenAI or Pinecone calls."""

import itertools

from ingest import CHUNK_OVERLAP, CHUNK_SIZE, Page, chunk_pages, make_chunk_id

KAN = "KAN Kolmogorov–Arnold Networks.pdf"  # contains an en dash (non-ASCII)


def _long_text(words: int = 600) -> str:
    return " ".join(f"word{i}" for i in range(words))


def test_empty_input_returns_no_chunks():
    assert chunk_pages([]) == []


def test_short_page_is_a_single_chunk():
    chunks = chunk_pages(
        [Page(source="a.pdf", page=3, text="Attention is all you need.")]
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Attention is all you need."
    assert chunks[0].chunk_index == 0


def test_long_page_splits_within_size_and_overlaps():
    chunks = chunk_pages([Page(source="a.pdf", page=1, text=_long_text())])

    assert len(chunks) > 1
    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)
    # Every word is unique, so if the next chunk's first word sits in the tail of
    # the previous chunk, the two chunks overlap.
    for previous, current in itertools.pairwise(chunks):
        assert current.text.split()[0] in previous.text[-CHUNK_OVERLAP:]


def test_metadata_is_preserved():
    pages = [Page(source=KAN, page=7, text=_long_text())]

    for chunk in chunk_pages(pages):
        assert chunk.source == KAN
        assert chunk.page == 7
        assert chunk.text in pages[0].text


def test_chunk_index_restarts_per_page():
    pages = [
        Page(source="a.pdf", page=1, text=_long_text()),
        Page(source="a.pdf", page=2, text=_long_text()),
    ]
    chunks = chunk_pages(pages)

    for page_number in (1, 2):
        indexes = [c.chunk_index for c in chunks if c.page == page_number]
        assert indexes == list(range(len(indexes)))


def test_ids_are_deterministic_unique_and_ascii():
    pages = [
        Page(source=KAN, page=1, text=_long_text()),
        Page(source=KAN, page=2, text=_long_text()),
        Page(source="Attention Is All You Need.pdf", page=1, text=_long_text()),
    ]
    first = [c.id for c in chunk_pages(pages)]
    second = [c.id for c in chunk_pages(pages)]

    assert first == second
    assert len(set(first)) == len(first)
    assert all(chunk_id.isascii() for chunk_id in first)


def test_chunk_id_format():
    chunk_id = make_chunk_id(KAN, page=4, chunk_index=2)

    assert chunk_id.startswith("kan-kolmogorov-arnold-networks-")
    assert chunk_id.endswith("-p4-c2")


def test_filenames_that_slugify_alike_get_distinct_ids():
    assert make_chunk_id("a b.pdf", 1, 0) != make_chunk_id("a-b.pdf", 1, 0)
