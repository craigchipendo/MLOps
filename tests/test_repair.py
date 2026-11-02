import pandas as pd

from newsclf.data.repair import find_collapsed_classes, repair_collapsed_titles, title_from_url


def test_title_from_url_strips_ids_and_extensions():
    url = "https://www.reuters.com/article/us-global-oil/oil-prices-rise-head-for-weekly-gain-idUSKCN25A04M"
    assert title_from_url(url) == "oil prices rise head for weekly gain"


def test_title_from_url_handles_html_and_dates():
    url = "https://www.cnbc.com/2020/08/11/boeing-order-cancellations-outpace-new-sales.html"
    assert title_from_url(url) == "boeing order cancellations outpace new sales"
    url = "https://example.com/news/exxon-chevron-shares-fall-2020-08-07"
    assert title_from_url(url) == "exxon chevron shares fall"


def test_title_from_url_returns_none_without_slug():
    assert title_from_url("https://www.bbc.com/news/business-53732395") is None
    assert title_from_url(None) is None


def test_find_collapsed_classes(synthetic_df):
    assert list(find_collapsed_classes(synthetic_df)) == ["BUSINESS"]


def test_repair_from_slug(synthetic_df):
    repaired, report = repair_collapsed_titles(synthetic_df)
    business = repaired[repaired.topic == "BUSINESS"]
    assert report["from_slug"] == 120 and report["dropped"] == 0
    assert business.title.nunique() == len(business)
    assert (business.title_source == "url_slug").all()


def test_repair_prefers_reference(synthetic_df):
    links = synthetic_df[synthetic_df.topic == "BUSINESS"].link.head(10)
    reference = pd.DataFrame({"link": links, "title": [f"Trusted headline {i}" for i in range(10)]})
    repaired, report = repair_collapsed_titles(synthetic_df, reference=reference)
    assert report["from_reference"] == 10
    assert report["from_slug"] == 110
    assert (repaired.title_source == "reference").sum() == 10
