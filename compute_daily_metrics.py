"""
Compute daily volume and intraday price volatility for all Polymarket contracts.
Outputs one row per outcome token per day for Aug 7 - Oct 7, 2025.
"""
import polars as pl
from datetime import datetime
from poly_utils.utils import get_markets

# --- 1. Load and filter trades to Aug 7 - Oct 7, 2025 ---
print("Loading trades...")
trades = (
    pl.scan_csv("processed/trades.csv")
    # Pre-filter on raw string (ISO format is lexicographically sortable)
    # to avoid parsing all 35GB of timestamps before filtering
    .filter(
        (pl.col("timestamp") >= "2025-08-07") &
        (pl.col("timestamp") < "2025-10-08")
    )
    .filter(pl.col("market_id").is_not_null())
    .with_columns(pl.col("timestamp").str.to_datetime())
    .collect(streaming=True)
)
print(f"Loaded {len(trades):,} trades in Aug 7 - Oct 7, 2025")

# --- 2. Add date column ---
trades = trades.with_columns(pl.col("timestamp").dt.date().alias("date"))

# --- 3. Compute token-level daily metrics ---
print("Computing daily metrics...")
daily_metrics = (
    trades
    .group_by(["market_id", "nonusdc_side", "date"])
    .agg([
        pl.col("usd_amount").sum().alias("daily_volume_usd"),
        pl.col("usd_amount").count().alias("trade_count"),
        pl.col("price").std().alias("price_std"),
        (pl.col("price").max() - pl.col("price").min()).alias("price_range"),
        pl.col("price").mean().alias("avg_price"),
        pl.col("price").min().alias("price_low"),
        pl.col("price").max().alias("price_high"),
    ])
    .sort(["market_id", "date", "nonusdc_side"])
)

# --- 4. Enrich with market metadata ---
print("Loading market metadata...")
markets = get_markets().rename({"id": "market_id"})

# nonusdc_side is "token1" or "token2", so map to the answer label
output = daily_metrics.join(
    markets.select(["market_id", "question", "answer1", "answer2", "ticker", "category"]),
    on="market_id",
    how="left",
)

# Map nonusdc_side ("token1"/"token2") to the outcome label ("Yes"/"No"/etc.)
output = output.with_columns(
    pl.when(pl.col("nonusdc_side") == "token1")
    .then(pl.col("answer1"))
    .when(pl.col("nonusdc_side") == "token2")
    .then(pl.col("answer2"))
    .otherwise(pl.lit("Unknown"))
    .alias("outcome")
)

# Final column selection and ordering
output = output.select([
    "date",
    "market_id",
    "question",
    "ticker",
    "category",
    "outcome",
    "nonusdc_side",
    "daily_volume_usd",
    "trade_count",
    "price_std",
    "price_range",
    "avg_price",
    "price_low",
    "price_high",
])

# --- 5. Write output ---
output.write_csv("aug7_oct7_2025_daily_metrics.csv")
print(f"Wrote {len(output):,} rows to aug7_oct7_2025_daily_metrics.csv")

# Summary stats
print(f"\nSummary:")
print(f"  Date range: {output['date'].min()} to {output['date'].max()}")
print(f"  Unique markets: {output['market_id'].n_unique()}")
print(f"  Unique days: {output['date'].n_unique()}")
