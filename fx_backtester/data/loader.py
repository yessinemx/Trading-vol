"""Market data loading and preprocessing from Parquet files."""

import pandas as pd
from pathlib import Path


class MarketDataLoader:
    """Loads and pivots stacked market data from Parquet files."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load_spot(self) -> pd.DataFrame:
        path = self._find("spot")
        df = pd.read_parquet(path)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["MID_PRICE"] = pd.to_numeric(df["MID_PRICE"], errors="coerce")
        return df.set_index("DATE").sort_index()

    def load_forward_curve(self) -> pd.DataFrame:
        path = self._find("forwardcurve")
        df = pd.read_parquet(path)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.rename(columns={"Maturity": "MATURITY"})
        df["MATURITY"] = pd.to_datetime(df["MATURITY"])
        df["MID_PRICE"] = pd.to_numeric(df["MID_PRICE"], errors="coerce")
        # filter sentinel values and convert pips → price units
        df = df[df["MID_PRICE"].abs() < 9000].copy()
        df["MID_PRICE"] = df["MID_PRICE"] / 10_000
        return df.sort_values(["DATE", "MATURITY"])

    def load_interest_rates(self) -> pd.DataFrame:
        """Load and combine domestic + foreign deposit curves (curve_id 1=domestic, 2=foreign)."""
        paths = sorted(self.data_dir.glob("*depocurve*.parquet"))
        frames = [pd.read_parquet(p) for p in paths]
        for i, frame in enumerate(frames):
            frame["curve_id"] = i + 1
        df = pd.concat(frames, ignore_index=True)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.rename(columns={"Maturity": "MATURITY"})
        df["MATURITY"] = pd.to_datetime(df["MATURITY"])
        df["MID_PRICE"] = pd.to_numeric(df["MID_PRICE"], errors="coerce")
        # rates are in % annual → convert to decimal
        df["MID_PRICE"] = df["MID_PRICE"] / 100
        return df.sort_values(["curve_id", "DATE", "MATURITY"])

    def load_vol_surface(self) -> pd.DataFrame:
        path = self._find("volatilitysurface")
        df = pd.read_parquet(path)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["MATURITY"] = pd.to_datetime(df["MATURITY"])
        df["MID_PRICE"] = pd.to_numeric(df["MID_PRICE"], errors="coerce")
        df["MID_STRIKE"] = pd.to_numeric(df["MID_STRIKE"], errors="coerce")
        df = df.dropna(subset=["MID_PRICE", "MID_STRIKE"])
        return df.sort_values(["DATE", "MATURITY", "MID_STRIKE"])

    def _find(self, keyword: str) -> Path:
        matches = list(self.data_dir.glob(f"*{keyword}*.parquet"))
        if not matches:
            raise FileNotFoundError(f"No parquet file matching '*{keyword}*' in {self.data_dir}")
        return matches[0]
