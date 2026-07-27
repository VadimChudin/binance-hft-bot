"""Configuration loading and typed access."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class ScannerConfig:
    refresh_sec: int = 300
    top_n: int = 8
    min_quote_volume: float = 50_000_000
    max_spread_pct: float = 0.0005
    min_volatility_pct: float = 0.0015
    perpetual_only: bool = True


@dataclass
class FeedConfig:
    kline_interval: str = "1m"
    depth_levels: int = 20


@dataclass
class StrategyConfig:
    enabled: bool = True
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatorConfig:
    entry_threshold: float = 0.5


@dataclass
class RiskConfig:
    paper_equity: float = 1000.0
    leverage: int = 5
    risk_per_trade_pct: float = 0.02
    stop_loss_pct: float = 0.004
    take_profit_pct: float = 0.006
    max_holding_sec: int = 300
    max_open_positions: int = 3
    max_daily_loss: float = 50.0
    cooldown_sec: int = 60
    taker_fee_pct: float = 0.0004


@dataclass
class EngineConfig:
    tick_sec: float = 1.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    dir: str = "logs"


@dataclass
class Credentials:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False

    @property
    def has_keys(self) -> bool:
        return bool(self.api_key and self.api_secret)


@dataclass
class Config:
    mode: str = "paper"
    quote_asset: str = "USDT"
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    feed: FeedConfig = field(default_factory=FeedConfig)
    strategies: dict[str, StrategyConfig] = field(default_factory=dict)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    credentials: Credentials = field(default_factory=Credentials)

    @property
    def is_live(self) -> bool:
        return self.mode.lower() == "live"


def _strategies_from_dict(raw: dict[str, Any]) -> dict[str, StrategyConfig]:
    out: dict[str, StrategyConfig] = {}
    for name, cfg in (raw or {}).items():
        cfg = dict(cfg or {})
        enabled = bool(cfg.pop("enabled", True))
        weight = float(cfg.pop("weight", 1.0))
        out[name] = StrategyConfig(enabled=enabled, weight=weight, params=cfg)
    return out


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load YAML config and merge credentials from the environment/.env."""
    load_dotenv()

    path = Path(path)
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    creds = Credentials(
        api_key=os.getenv("BINANCE_API_KEY", "").strip(),
        api_secret=os.getenv("BINANCE_API_SECRET", "").strip(),
        testnet=os.getenv("BINANCE_TESTNET", "false").strip().lower() == "true",
    )

    return Config(
        mode=str(raw.get("mode", "paper")),
        quote_asset=str(raw.get("quote_asset", "USDT")),
        scanner=ScannerConfig(**(raw.get("scanner") or {})),
        feed=FeedConfig(**(raw.get("feed") or {})),
        strategies=_strategies_from_dict(raw.get("strategies") or {}),
        aggregator=AggregatorConfig(**(raw.get("aggregator") or {})),
        risk=RiskConfig(**(raw.get("risk") or {})),
        engine=EngineConfig(**(raw.get("engine") or {})),
        logging=LoggingConfig(**(raw.get("logging") or {})),
        credentials=creds,
    )
