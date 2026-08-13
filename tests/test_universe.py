"""Offline tests for universe parsing/filtering (no network).

Uses synthetic pipe-delimited text that mirrors the NASDAQ Trader file format
(header row + trailing pipe) to exercise the parser and the common-stock filter.
"""

import pandas as pd

from stonks.universe import _parse_combined, filter_common_stocks

# Synthetic nasdaqlisted.txt (header + trailing pipe, like the real file).
_NASDAQ_TEXT = "\n".join(
    [
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares|",
        "AAPL|Apple Inc. Common Stock|Q|N|N|100|N|N|",
        "SPY|SPDR S&P 500 ETF Trust|G|N|N|100|Y|N|",
        "ZTEST|Test Issue Inc. Common Stock|Q|Y|N|100|N|N|",
        "ZSPAC|Foo Acquisition Corp. Common Stock|Q|N|N|100|N|N|",
        "ZWARR|Foo Inc. Warrant|Q|N|N|100|N|N|",
        "ZPFD|Foo Preferred Stock|Q|N|N|100|N|N|",
        "",
    ]
)

# Synthetic otherlisted.txt (NYSE/other listings).
_OTHER_TEXT = "\n".join(
    [
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol|",
        "BRK.B|Berkshire Hathaway Inc. Common Stock|N|BRK.B|N|5|N|BRK.B|",
        "F|Ford Motor Company Common Stock|N|F|N|100|N|F|",
        "ZFD|Foo Closed End Fund|N|ZFD|N|100|N|ZFD|",
        "",
    ]
)


class TestParse:
    def test_parse_combines_both_files(self) -> None:
        combined = _parse_combined(_NASDAQ_TEXT, _OTHER_TEXT)
        # 6 nasdaq + 3 other data rows (headers dropped).
        assert len(combined) == 9
        assert list(combined.columns) == ["ticker", "name", "exchange", "is_etf", "is_test"]

    def test_parse_exchange_mapping(self) -> None:
        combined = _parse_combined(_NASDAQ_TEXT, _OTHER_TEXT)
        nyse = combined[combined["ticker"] == "F"].iloc[0]
        assert nyse["exchange"] == "NYSE"
        nasdaq = combined[combined["ticker"] == "AAPL"].iloc[0]
        assert nasdaq["exchange"] == "NASDAQ"

    def test_header_row_is_not_treated_as_data(self) -> None:
        combined = _parse_combined(_NASDAQ_TEXT, _OTHER_TEXT)
        assert "SYMBOL" not in set(combined["ticker"])
        assert "ACT SYMBOL" not in set(combined["ticker"])


class TestFilter:
    def test_keeps_common_stocks(self) -> None:
        combined = _parse_combined(_NASDAQ_TEXT, _OTHER_TEXT)
        kept = filter_common_stocks(combined)
        tickers = set(kept["ticker"])
        # Common stocks on NASDAQ and NYSE are kept.
        assert "AAPL" in tickers
        assert "BRK.B" in tickers  # dot in ticker is allowed
        assert "F" in tickers

    def test_drops_etfs_test_issues_exotics(self) -> None:
        combined = _parse_combined(_NASDAQ_TEXT, _OTHER_TEXT)
        tickers = set(filter_common_stocks(combined)["ticker"])
        assert "SPY" not in tickers      # ETF
        assert "ZTEST" not in tickers    # test issue
        assert "ZSPAC" not in tickers    # SPAC ("Acquisition Corp")
        assert "ZWARR" not in tickers    # warrant (no "common stock")
        assert "ZPFD" not in tickers     # preferred (no "common stock")
        assert "ZFD" not in tickers      # fund (no "common stock")

    def test_result_is_deduped_sorted_clean(self) -> None:
        combined = _parse_combined(_NASDAQ_TEXT, _OTHER_TEXT)
        kept = filter_common_stocks(combined)
        assert list(kept.columns) == ["ticker", "name", "exchange"]
        # Sorted ascending by ticker.
        assert kept["ticker"].is_monotonic_increasing
        # No duplicates.
        assert kept["ticker"].is_unique
        # Exactly the three common stocks.
        assert kept["ticker"].tolist() == ["AAPL", "BRK.B", "F"]

    def test_filter_on_synthetic_frame_directly(self) -> None:
        # filter_common_stocks is pure: feed a hand-built frame.
        combined = pd.DataFrame(
            [
                {"ticker": "A", "name": "A Common Stock", "exchange": "NASDAQ", "is_etf": "N", "is_test": "N"},
                {"ticker": "B", "name": "B ETF", "exchange": "NASDAQ", "is_etf": "Y", "is_test": "N"},
                {"ticker": "C", "name": "C Common Stock", "exchange": "NYSE", "is_etf": "N", "is_test": "Y"},
                {"ticker": "D", "name": "D Common Stock", "exchange": "NYSE", "is_etf": "N", "is_test": "N"},
            ]
        )
        kept = filter_common_stocks(combined)
        assert kept["ticker"].tolist() == ["A", "D"]
