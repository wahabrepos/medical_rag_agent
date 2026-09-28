from medrag_ingest.medlineplus import chunks_of, summary_blocks

SUMMARY = (
    "<h3>What is diabetes?</h3><p>Diabetes means your blood glucose is too high.</p>"
    "<h3>What are the symptoms of diabetes?</h3><p>The symptoms of diabetes may include:</p>"
    "<ul><li>Feeling very thirsty</li><li>Losing weight without trying</li></ul>"
    "<p>Symptoms may vary by type.</p>"
    "<p>NIH: National Institute of Diabetes and Digestive and Kidney Diseases</p>"
)


def test_blocks_mark_headings_items_and_drop_credit_lines() -> None:
    blocks = summary_blocks(SUMMARY)
    assert blocks[0] == ("heading", "What is diabetes?")
    assert ("item", "Feeling very thirsty") in blocks
    assert not any(t.startswith("NIH:") for _, t in blocks)


def test_a_list_stays_with_its_lead_in_and_ends_its_chunk() -> None:
    chunks = chunks_of(summary_blocks(SUMMARY))
    assert chunks == [
        "What is diabetes? Diabetes means your blood glucose is too high.",
        "What are the symptoms of diabetes? The symptoms of diabetes may include: "
        "Feeling very thirsty. Losing weight without trying.",
        "What are the symptoms of diabetes? Symptoms may vary by type.",
    ]


def test_a_split_list_repeats_its_lead_in() -> None:
    items = "".join(f"<li>symptom number {i} here</li>" for i in range(6))
    html = f"<h3>Symptoms</h3><p>Symptoms may include:</p><ul>{items}</ul>"
    chunks = chunks_of(summary_blocks(html), max_words=12)
    assert len(chunks) > 1
    assert all(c.startswith("Symptoms Symptoms may include:") for c in chunks)
