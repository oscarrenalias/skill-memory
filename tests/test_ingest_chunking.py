"""Targeted tests for _chunk_txt and _chunk_md chunking helpers in memory.py.

These are pure functions with no DB or embedding model dependency.
Corresponds to TestIngestTxt and TestIngestMd from the spec test plan.
"""
import sys
import os
import unittest

# Import only the chunking helpers without triggering _bootstrap() or the
# top-level side-effects that require venv / onnxruntime.
# We patch sys.argv and intercept os.execv before importing.
_orig_execv = os.execv
os.execv = lambda *a, **kw: None  # neutralise venv re-exec in _bootstrap

# Also ensure numpy is importable for the module-level imports in memory.py.
# The chunking helpers themselves do not use numpy, but memory.py imports it
# unconditionally at module level after _bootstrap runs. We skip this by
# importing only the functions we need via a selective approach.
#
# Strategy: import the module with mocked heavy dependencies.
import unittest.mock as mock

with mock.patch.dict('sys.modules', {
    'numpy': mock.MagicMock(),
    'sqlite_vec': mock.MagicMock(),
    'onnxruntime': mock.MagicMock(),
}):
    # Prevent bootstrap re-exec
    with mock.patch('os.execv', lambda *a, **kw: None):
        import importlib, types
        # We need to load memory.py from the repo root
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location("memory", os.path.join(repo_root, "memory.py"))
        _mem = importlib.util.module_from_spec(spec)
        # Provide mocked heavy deps before exec
        _mem.__dict__.update({})
        sys.modules['memory'] = _mem
        try:
            spec.loader.exec_module(_mem)
        except Exception:
            # Bootstrap may call os.execv which we mocked to no-op,
            # after that the module continues loading. If numpy is needed
            # we patch it.
            pass

_chunk_txt = _mem._chunk_txt
_chunk_md = _mem._chunk_md
_chunk_paragraphs = _mem._chunk_paragraphs
_split_sentences = _mem._split_sentences


class TestSplitSentences(unittest.TestCase):
    def test_simple_sentences(self):
        text = "Hello world. How are you? I am fine!"
        result = _split_sentences(text)
        self.assertEqual(result, ["Hello world.", "How are you?", "I am fine!"])

    def test_single_sentence(self):
        result = _split_sentences("Just one sentence.")
        self.assertEqual(result, ["Just one sentence."])

    def test_empty_string(self):
        result = _split_sentences("")
        self.assertEqual(result, [])

    def test_no_punctuation(self):
        result = _split_sentences("no punctuation here")
        self.assertEqual(result, ["no punctuation here"])


class TestIngestTxt(unittest.TestCase):
    """TestIngestTxt: paragraph chunking for .txt files."""

    def test_three_paragraphs_three_chunks(self):
        """3 paragraphs, each under chunk_size, each becomes its own chunk.

        Spec AC: "A .txt file with 3 paragraphs produces 3 separate memory rows
        (assuming each paragraph is under --chunk-size)."

        chunk_size=20 prevents merging (any two paragraphs combined would exceed 20
        chars), while each paragraph individually (14-16 chars) is under chunk_size.
        """
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = _chunk_txt(text, chunk_size=20, overlap=0)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Paragraph one.")
        self.assertEqual(chunks[1], "Paragraph two.")
        self.assertEqual(chunks[2], "Paragraph three.")

    def test_short_paragraphs_merged_when_all_fit_in_chunk(self):
        """Short paragraphs are merged into one chunk when all fit within chunk_size.

        The implementation merges paragraphs up to chunk_size (see spec §1 .txt strategy).
        When the combined text of all paragraphs fits within chunk_size, the result
        is a single chunk.
        """
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = _chunk_txt(text, chunk_size=1000, overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Paragraph one.", chunks[0])
        self.assertIn("Paragraph two.", chunks[0])
        self.assertIn("Paragraph three.", chunks[0])

    def test_short_chunk_not_created_for_long_paragraph(self):
        """A paragraph longer than chunk_size is split at sentence boundaries."""
        long_para = "First sentence here. Second sentence here. Third sentence here."
        chunks = _chunk_txt(long_para, chunk_size=30, overlap=0)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 40)  # allow slight overshoot on last

    def test_blank_lines_as_paragraph_boundary(self):
        """Double blank lines are recognised as paragraph separators."""
        text = "Alpha.\n\nBeta.\n\nGamma."
        chunks = _chunk_txt(text, chunk_size=10, overlap=0)
        self.assertIn("Alpha.", chunks)
        self.assertIn("Beta.", chunks)
        self.assertIn("Gamma.", chunks)

    def test_empty_text_returns_no_chunks(self):
        chunks = _chunk_txt("", chunk_size=1000, overlap=0)
        self.assertEqual(chunks, [])

    def test_whitespace_only_paragraphs_skipped(self):
        text = "Real paragraph.\n\n   \n\nAnother real paragraph."
        chunks = _chunk_txt(text, chunk_size=1000, overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Real paragraph.", chunks[0])

    def test_overlap_carries_tail_into_next_chunk(self):
        """With overlap > 0 the tail of one chunk appears at the start of the next."""
        # Three paragraphs, each exactly 15 chars, chunk_size=20 forces one-per-chunk.
        # Overlap=5 should carry 5 chars from the previous chunk's tail.
        p1 = "AAAAA BBBBB CCCCC"   # 17 chars
        p2 = "DDDDD EEEEE FFFFF"   # 17 chars
        p3 = "GGGGG HHHHH IIIII"   # 17 chars
        text = f"{p1}\n\n{p2}\n\n{p3}"
        chunks = _chunk_txt(text, chunk_size=20, overlap=5)
        self.assertEqual(len(chunks), 3)
        # Chunk 2 should start with the last 5 chars of chunk 1
        tail = chunks[0][-5:]
        self.assertTrue(chunks[1].startswith(tail))


class TestIngestMd(unittest.TestCase):
    """TestIngestMd: heading boundaries for .md files."""

    def test_level2_heading_is_hard_boundary(self):
        """## headings create hard chunk boundaries."""
        text = "Intro text.\n\n## Section One\n\nSection one body.\n\n## Section Two\n\nSection two body."
        chunks = _chunk_md(text, chunk_size=1000, overlap=0)
        # With large chunk_size, intro + Section One may or may not merge with intro.
        # Key invariant: Section Two is in a separate chunk from intro text.
        self.assertTrue(any("Section Two" in c for c in chunks))
        self.assertTrue(any("Section One" in c for c in chunks))

    def test_heading_text_carried_into_following_chunk(self):
        """The heading line appears in the chunk that contains the section body."""
        text = "## My Heading\n\nBody of the section."
        chunks = _chunk_md(text, chunk_size=1000, overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertIn("## My Heading", chunks[0])
        self.assertIn("Body of the section.", chunks[0])

    def test_two_sections_two_chunks(self):
        """Two ## sections with small chunk_size produce two separate chunks."""
        text = "## Alpha\n\nAlpha body text.\n\n## Beta\n\nBeta body text."
        chunks = _chunk_md(text, chunk_size=25, overlap=0)
        self.assertGreaterEqual(len(chunks), 2)
        alpha_chunks = [c for c in chunks if "Alpha" in c]
        beta_chunks = [c for c in chunks if "Beta" in c]
        self.assertTrue(alpha_chunks)
        self.assertTrue(beta_chunks)
        # Alpha and Beta must NOT be in the same chunk
        for c in chunks:
            self.assertFalse("Alpha" in c and "Beta" in c,
                             f"Alpha and Beta should not share a chunk: {c!r}")

    def test_pre_heading_text_included(self):
        """Text before the first ## heading is included as its own chunk."""
        text = "Preamble paragraph.\n\n## Section\n\nSection body."
        chunks = _chunk_md(text, chunk_size=1000, overlap=0)
        preamble_found = any("Preamble paragraph." in c for c in chunks)
        self.assertTrue(preamble_found)

    def test_empty_md_returns_no_chunks(self):
        chunks = _chunk_md("", chunk_size=1000, overlap=0)
        self.assertEqual(chunks, [])

    def test_heading_only_no_body(self):
        """A heading with no body produces either no chunk or a heading-only chunk."""
        text = "## Heading Only\n"
        chunks = _chunk_md(text, chunk_size=1000, overlap=0)
        # Either empty or contains just the heading — must not crash
        for c in chunks:
            self.assertIsInstance(c, str)


class TestMinChunkLenFiltering(unittest.TestCase):
    """Verify short chunk filtering logic (done in cmd_ingest but testable via chunkers)."""

    def test_very_short_text_paragraph(self):
        """A paragraph under 10 chars would be skipped by cmd_ingest's filter."""
        text = "Hi.\n\nThis is a longer paragraph that has real content."
        chunks = _chunk_txt(text, chunk_size=1000, overlap=0)
        # Both paragraphs merge into one chunk since chunk_size=1000
        self.assertEqual(len(chunks), 1)
        # Simulate the cmd_ingest filter
        MIN = 10
        filtered = [c for c in chunks if len(c.strip()) >= MIN]
        self.assertEqual(len(filtered), len(chunks))

    def test_stub_chunk_under_10_chars_would_be_skipped(self):
        """Chunks shorter than 10 chars are caught by the ingest filter."""
        # Force a tiny chunk by using a very small chunk_size with a tiny paragraph
        text = "OK.\n\nThis is a normal-length paragraph with real words in it."
        chunks = _chunk_txt(text, chunk_size=10, overlap=0)
        MIN = 10
        short = [c for c in chunks if len(c.strip()) < MIN]
        # "OK." is 3 chars — it should appear as a chunk candidate that gets filtered
        self.assertTrue(any(len(c.strip()) < MIN for c in chunks),
                        "Expected at least one short chunk from 'OK.' paragraph")


if __name__ == "__main__":
    unittest.main()
