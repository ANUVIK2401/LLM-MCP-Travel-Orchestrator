from listing_parser import extract_url, parse_assistant_response, parse_row


def test_extract_url_supports_markdown_links():
    text, url = extract_url("[Open listing](https://www.airbnb.com/rooms/12345)")
    assert text == "Open listing"
    assert url == "https://www.airbnb.com/rooms/12345"


def test_parse_row_extracts_listing_fields():
    row = parse_row(
        "Canal Loft | $145/night | 4.91 | Bright loft near the center | https://www.airbnb.com/rooms/555"
    )

    assert row is not None
    assert row.name == "Canal Loft"
    assert row.price == "$145/night"
    assert row.rating == "4.91"
    assert row.rating_num == 4.91
    assert row.link == "https://www.airbnb.com/rooms/555"


def test_parse_assistant_response_deduplicates_rows_and_collects_notes_and_tips():
    response = """
    Here are the best matches for your budget.
    Name | Price | Rating | Description | URL
    Canal Loft | $145/night | 4.91 | Bright loft near the center | https://www.airbnb.com/rooms/555
    Canal Loft | $145/night | 4.91 | Bright loft near the center | https://www.airbnb.com/rooms/555
    - Try widening the radius if you want more inventory.
    """

    parsed = parse_assistant_response(response)

    assert parsed.notes == ["Here are the best matches for your budget."]
    assert len(parsed.listings) == 1
    assert parsed.listings[0].name == "Canal Loft"
    assert parsed.tips == ["Try widening the radius if you want more inventory."]
