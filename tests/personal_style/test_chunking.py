from personal_style_pl.textsplit import Chunk, chunk_document


PL = (" Zdanie pierwsze ma kilka słów. Zdanie drugie jest tutaj. "
      "Trzecie zdanie również. Czwarte zdanie tutaj jest. "
      "Piąte zdanie mamy. Szóste zdanie też. Siódme zdanie krótkie. "
      "Ósme zdanie ostatnie. Dziewiąte zdanie dodatkowe. Dziesiąte zdanie tu.")


def test_chunking_preserves_diacritics():
    chunks = chunk_document(PL + " Łódź, gęś, źdźbło, ćma.", doc_id="d0",
                            chunk_sentences=8, min_chunk_tokens=5)
    joined = " ".join(c.text for c in chunks)
    for ch in "łęśćź":
        assert ch in joined


def test_chunk_groups_by_sentence_count():
    chunks = chunk_document(PL, doc_id="d0", chunk_sentences=4, min_chunk_tokens=1)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == "d0" for c in chunks)
    assert chunks[0].sentence_count == 4


def test_short_chunks_merge_to_min_tokens():
    chunks = chunk_document(PL, doc_id="d0", chunk_sentences=1, min_chunk_tokens=120)
    # whole thing is < 120 tokens -> a single (under-min) chunk, flagged
    assert len(chunks) == 1
    assert chunks[0].under_min_tokens is True


def test_empty_text_returns_no_chunks():
    assert chunk_document("   ", doc_id="d0") == []
