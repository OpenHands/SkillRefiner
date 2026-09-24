from skill_refiner.summarize import NoSummaryTraceSummarizer


def test_no_summary_summarizer_is_constructible():
    assert NoSummaryTraceSummarizer() is not None
