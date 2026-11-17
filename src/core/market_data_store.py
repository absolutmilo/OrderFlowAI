"""Local SQLite storage for market features and forward-outcome training data."""

import hashlib
import json
import math
import os
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
        max_database_size_bytes: int = 1024 * 1024 * 1024,
        retention_target_fraction: float = 0.90,
    ):
        if max_database_size_bytes <= 0:
            raise ValueError("The maximum database size must be greater than zero.")
        if not 0 < retention_target_fraction < 1:
            raise ValueError("The database retention target must be between zero and one.")
        self.database_path = database_path
        self.label_horizons_seconds = label_horizons_seconds
        self.min_move_bps = min_move_bps
        self.label_tolerance_seconds = label_tolerance_seconds
        self.max_database_size_bytes = max_database_size_bytes
        self.retention_target_fraction = retention_target_fraction
        self.connection: Optional[sqlite3.Connection] = None
        self._last_profile_prune_at_ms = 0

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA wal_autocheckpoint=1000")
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
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orderflow_trade_profile (
                bucket_start_ms INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                market_source TEXT NOT NULL,
                price REAL NOT NULL,
                buy_volume_btc REAL NOT NULL DEFAULT 0,
                sell_volume_btc REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (bucket_start_ms, symbol, market_source, price)
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_orderflow_profile_source_time "
            "ON orderflow_trade_profile(market_source, symbol, bucket_start_ms)"
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
        self._enforce_size_limit()

    def record_profile_trade(
        self,
        timestamp_ms: int,
        symbol: str,
        market_source: str,
        price: float,
        size_btc: float,
        side: str,
    ) -> None:
        if (
            timestamp_ms < 0
            or not math.isfinite(price)
            or not math.isfinite(size_btc)
            or price <= 0
            or size_btc <= 0
        ):
            raise ValueError("El timestamp, el precio y el tamaño deben ser positivos.")
        if side not in {"BUY", "SELL"}:
            raise ValueError("El lado del trade debe ser BUY o SELL.")

        bucket_start_ms = timestamp_ms - timestamp_ms % 60_000
        volume_column = "buy_volume_btc" if side == "BUY" else "sell_volume_btc"
        connection = self._connection()
        with connection:
            connection.execute(
                f"""
                INSERT INTO orderflow_trade_profile
                    (bucket_start_ms, symbol, market_source, price, {volume_column})
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (bucket_start_ms, symbol, market_source, price)
                DO UPDATE SET {volume_column} =
                    {volume_column} + excluded.{volume_column}
                """,
                (bucket_start_ms, symbol, market_source, round(price, 2), size_btc),
            )

            if timestamp_ms - self._last_profile_prune_at_ms >= 60_000:
                connection.execute(
                    "DELETE FROM orderflow_trade_profile WHERE bucket_start_ms < ?",
                    (timestamp_ms - 25 * 60 * 60 * 1000,),
                )
                self._last_profile_prune_at_ms = timestamp_ms

    def get_orderflow_profile(
        self,
        symbol: str,
        market_source: str,
        profile_start_ms: int,
        session_start_ms: int,
        end_ms: int,
    ) -> Dict[str, object]:
        connection = self._connection()
        levels = connection.execute(
            """
            SELECT
                price,
                SUM(buy_volume_btc) AS buy_volume_btc,
                SUM(sell_volume_btc) AS sell_volume_btc
            FROM orderflow_trade_profile
            WHERE symbol = ? AND market_source = ?
              AND bucket_start_ms >= ? AND bucket_start_ms <= ?
            GROUP BY price
            ORDER BY price
            """,
            (symbol, market_source, profile_start_ms, end_ms),
        ).fetchall()
        profile_coverage = connection.execute(
            """
            SELECT
                MIN(bucket_start_ms),
                MAX(bucket_start_ms),
                COUNT(DISTINCT bucket_start_ms)
            FROM orderflow_trade_profile
            WHERE symbol = ? AND market_source = ?
              AND bucket_start_ms >= ? AND bucket_start_ms <= ?
            """,
            (symbol, market_source, profile_start_ms, end_ms),
        ).fetchone()
        session = connection.execute(
            """
            SELECT
                MIN(bucket_start_ms),
                MAX(bucket_start_ms),
                SUM(buy_volume_btc + sell_volume_btc),
                SUM(price * (buy_volume_btc + sell_volume_btc)),
                COUNT(DISTINCT bucket_start_ms)
            FROM orderflow_trade_profile
            WHERE symbol = ? AND market_source = ?
              AND bucket_start_ms >= ? AND bucket_start_ms <= ?
            """,
            (symbol, market_source, session_start_ms, end_ms),
        ).fetchone()
        current_price_row = connection.execute(
            """
            SELECT price FROM market_observations
            WHERE symbol = ? AND market_source = ? AND observed_at_ms <= ?
            ORDER BY observed_at_ms DESC, id DESC
            LIMIT 1
            """,
            (symbol, market_source, end_ms),
        ).fetchone()

        profile_levels = []
        for price, buy_volume_btc, sell_volume_btc in levels:
            total_volume_btc = buy_volume_btc + sell_volume_btc
            profile_levels.append({
                "price": price,
                "buy_volume_btc": buy_volume_btc,
                "sell_volume_btc": sell_volume_btc,
                "volume_btc": total_volume_btc,
                "buy_volume_usd": price * buy_volume_btc,
                "sell_volume_usd": price * sell_volume_btc,
                "delta_usd": price * (buy_volume_btc - sell_volume_btc),
                "volume_usd": price * total_volume_btc,
            })

        session_volume_btc = session[2] or 0.0
        session_volume_usd = session[3] or 0.0
        return {
            "levels": profile_levels,
            "profile_first_trade_ms": profile_coverage[0],
            "profile_last_trade_ms": profile_coverage[1],
            "profile_bucket_count": profile_coverage[2],
            "session_first_trade_ms": session[0],
            "session_last_trade_ms": session[1],
            "session_volume_btc": session_volume_btc,
            "session_volume_usd": session_volume_usd,
            "session_bucket_count": session[4],
            "session_vwap": (
                session_volume_usd / session_volume_btc
                if session_volume_btc > 0 else None
            ),
            "current_price": current_price_row[0] if current_price_row else None,
        }

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
        headroom_bytes = min(
            16 * 1024 * 1024,
            int(self.max_database_size_bytes * 0.1),
        )
        if self.storage_size_bytes() > self.max_database_size_bytes - headroom_bytes:
            self._enforce_size_limit(force=True)

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
        self._enforce_size_limit()

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

    def storage_size_bytes(self) -> int:
        return sum(
            os.path.getsize(path)
            for path in (
                self.database_path,
                Path(f"{self.database_path}-wal"),
                Path(f"{self.database_path}-shm"),
            )
            if path.exists()
        )

    def _enforce_size_limit(self, force: bool = False) -> None:
        connection = self._connection()
        target_size_bytes = int(
            self.max_database_size_bytes * self.retention_target_fraction
        )
        trigger_size_bytes = (
            target_size_bytes if force else self.max_database_size_bytes
        )
        if self.storage_size_bytes() <= trigger_size_bytes:
            return

        while self.storage_size_bytes() > target_size_bytes:
            row = connection.execute(
                "SELECT COUNT(*) FROM market_observations"
            ).fetchone()
            observation_count = int(row[0])
            if observation_count <= 1:
                raise sqlite3.DatabaseError(
                    "SQLite storage exceeds its configured size limit, and no market "
                    "observations can be pruned without removing the latest record."
                )

            current_size = max(1, self.storage_size_bytes())
            estimated_delete_count = max(
                1,
                math.ceil(
                    observation_count
                    * (current_size - target_size_bytes)
                    / current_size
                ),
            )
            delete_count = min(observation_count - 1, estimated_delete_count)
            with connection:
                connection.execute(
                    """
                    DELETE FROM market_observations
                    WHERE id IN (
                        SELECT id FROM market_observations
                        ORDER BY observed_at_ms ASC, id ASC
                        LIMIT ?
                    )
                    """,
                    (delete_count,),
                )
                connection.execute(
                    """
                    DELETE FROM configuration_snapshots
                    WHERE NOT EXISTS (
                        SELECT 1 FROM market_observations
                        WHERE market_observations.configuration_id =
                              configuration_snapshots.id
                    )
                    """
                )

            checkpoint = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if checkpoint is not None and checkpoint[0] != 0:
                raise sqlite3.OperationalError(
                    "Could not checkpoint SQLite before enforcing the storage limit."
                )
            connection.execute("VACUUM")
            checkpoint = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if checkpoint is not None and checkpoint[0] != 0:
                raise sqlite3.OperationalError(
                    "Could not checkpoint SQLite after enforcing the storage limit."
                )

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _connection(self) -> sqlite3.Connection:
        if self.connection is None:
            raise RuntimeError("Market data store has not been initialized.")
        return self.connection
