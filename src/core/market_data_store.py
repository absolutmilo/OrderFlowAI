"""Local SQLite storage for market features and forward-outcome training data."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple


class MarketDataStore:
    def __init__(
        self,
        database_path: Path,
        label_horizons_seconds: Tuple[int, ...],
        min_move_bps: float,
        label_tolerance_seconds: int = 2,
    ):
        self.database_path = database_path
        self.label_horizons_seconds = label_horizons_seconds
        self.min_move_bps = min_move_bps
        self.label_tolerance_seconds = label_tolerance_seconds
        self.connection: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS configuration_snapshots (
                id INTEGER PRIMARY KEY,
                recorded_at_ms INTEGER NOT NULL,
                config_hash TEXT NOT NULL UNIQUE,
                config_json TEXT NOT NULL
            )
            """
        )
        observation_columns = [
            "id INTEGER PRIMARY KEY",
            "observed_at_ms INTEGER NOT NULL",
            "symbol TEXT NOT NULL",
            "market_source TEXT NOT NULL DEFAULT 'legacy_combined'",
            "price REAL NOT NULL",
            "imbalance_direction INTEGER NOT NULL",
            "features_json TEXT NOT NULL",
            "heuristic_scores_json TEXT NOT NULL",
            "configuration_id INTEGER NOT NULL REFERENCES configuration_snapshots(id)",
        ]
        for horizon in self.label_horizons_seconds:
            observation_columns.extend((
                f"return_{horizon}s_bps REAL",
                f"label_{horizon}s TEXT",
            ))
        self.connection.execute(
            f"CREATE TABLE IF NOT EXISTS market_observations ({', '.join(observation_columns)})"
        )
        existing_columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(market_observations)")
        }
        for horizon in self.label_horizons_seconds:
            for column, column_type in (
                (f"return_{horizon}s_bps", "REAL"),
                (f"label_{horizon}s", "TEXT"),
            ):
                if column not in existing_columns:
                    self.connection.execute(
                        f"ALTER TABLE market_observations ADD COLUMN {column} {column_type}"
                    )
        if "market_source" not in existing_columns:
            self.connection.execute(
                "ALTER TABLE market_observations ADD COLUMN "
                "market_source TEXT NOT NULL DEFAULT 'legacy_combined'"
            )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_observations_pending "
            "ON market_observations(observed_at_ms)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_observations_source_time "
            "ON market_observations(market_source, observed_at_ms)"
        )
        self.connection.commit()

    def record_observation(
        self,
        observed_at_ms: int,
        symbol: str,
        price: float,
        cvd_usd: float,
        features: Dict[str, object],
        heuristic_scores: Dict[str, object],
        configuration: Dict[str, object],
        market_source: str = "legacy_combined",
    ) -> None:
        connection = self._connection()
        config_json = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        with connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO configuration_snapshots
                    (recorded_at_ms, config_hash, config_json)
                VALUES (?, ?, ?)
                """,
                (observed_at_ms, config_hash, config_json),
            )
            config_row = connection.execute(
                "SELECT id FROM configuration_snapshots WHERE config_hash = ?",
                (config_hash,),
            ).fetchone()
            if config_row is None:
                raise RuntimeError("Configuration snapshot was not persisted.")

            self._resolve_due_labels(connection, observed_at_ms, price, market_source)

            imbalance_direction = (cvd_usd > 0) - (cvd_usd < 0)
            columns = [
                "observed_at_ms", "symbol", "market_source", "price", "imbalance_direction",
                "features_json", "heuristic_scores_json", "configuration_id",
            ]
            values = [
                observed_at_ms,
                symbol,
                market_source,
                price,
                imbalance_direction,
                json.dumps(features, separators=(",", ":")),
                json.dumps(heuristic_scores, separators=(",", ":")),
                config_row[0],
            ]
            for horizon in self.label_horizons_seconds:
                columns.extend((f"return_{horizon}s_bps", f"label_{horizon}s"))
                values.extend((None, None))
            placeholders = ", ".join("?" for _ in values)
            connection.execute(
                f"INSERT INTO market_observations ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )

    def _resolve_due_labels(
        self,
        connection: sqlite3.Connection,
        observed_at_ms: int,
        current_price: float,
        market_source: str,
    ) -> None:
        due_conditions = " OR ".join(
            f"(return_{horizon}s_bps IS NULL AND observed_at_ms BETWEEN ? AND ?)"
            for horizon in self.label_horizons_seconds
        )
        due_cutoffs = [
            cutoff
            for horizon in self.label_horizons_seconds
            for cutoff in (
                observed_at_ms - horizon * 1000 - self.label_tolerance_seconds * 1000,
                observed_at_ms - horizon * 1000,
            )
        ]
        rows = connection.execute(
            f"""
            SELECT id, observed_at_ms, price, imbalance_direction,
                   {', '.join(f'return_{horizon}s_bps' for horizon in self.label_horizons_seconds)}
            FROM market_observations
            WHERE market_source = ? AND ({due_conditions})
            """,
            [market_source, *due_cutoffs],
        ).fetchall()

        for row in rows:
            observation_id, start_ms, start_price, direction = row[:4]
            prior_returns = row[4:]
            updates = []
            values = []
            for index, horizon in enumerate(self.label_horizons_seconds):
                target_ms = start_ms + horizon * 1000
                if (
                    prior_returns[index] is not None
                    or observed_at_ms < target_ms
                    or observed_at_ms > target_ms + self.label_tolerance_seconds * 1000
                ):
                    continue
                forward_return_bps = ((current_price - start_price) / start_price) * 10000.0
                if direction == 0 or abs(forward_return_bps) < self.min_move_bps:
                    label = "neutral"
                elif (forward_return_bps > 0) == (direction > 0):
                    label = "continuation"
                else:
                    label = "reversal"
                updates.extend((f"return_{horizon}s_bps = ?", f"label_{horizon}s = ?"))
                values.extend((forward_return_bps, label))
            if updates:
                connection.execute(
                    f"UPDATE market_observations SET {', '.join(updates)} WHERE id = ?",
                    (*values, observation_id),
                )

    def observation_count(self) -> int:
        row = self._connection().execute("SELECT COUNT(*) FROM market_observations").fetchone()
        return int(row[0])

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _connection(self) -> sqlite3.Connection:
        if self.connection is None:
            raise RuntimeError("Market data store has not been initialized.")
        return self.connection
