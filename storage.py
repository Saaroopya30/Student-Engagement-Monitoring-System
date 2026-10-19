from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List


class EngagementStorage:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "engagement_results.db"
        self.csv_path = self.output_dir / "engagement_results.csv"
        if self.db_path.exists():
            self.db_path.unlink()
        if self.csv_path.exists():
            self.csv_path.unlink()
        self.connection = sqlite3.connect(self.db_path)
        self._create_tables()
        self._csv_initialized = False

    def _create_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_results (
                frame_index INTEGER,
                timestamp_seconds REAL,
                track_id INTEGER,
                label TEXT,
                movement_score REAL,
                head_centered INTEGER,
                hand_raised INTEGER,
                upright_posture INTEGER,
                face_detected INTEGER,
                excessive_movement INTEGER
            )
            """
        )
        self.connection.commit()

    def write_records(self, records: List[Dict[str, object]]) -> None:
        if not records:
            return

        self.connection.executemany(
            """
            INSERT INTO frame_results (
                frame_index,
                timestamp_seconds,
                track_id,
                label,
                movement_score,
                head_centered,
                hand_raised,
                upright_posture,
                face_detected,
                excessive_movement
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["frame_index"],
                    row["timestamp_seconds"],
                    row["track_id"],
                    row["label"],
                    row["movement_score"],
                    row["head_centered"],
                    row["hand_raised"],
                    row["upright_posture"],
                    row["face_detected"],
                    row["excessive_movement"],
                )
                for row in records
            ],
        )
        self.connection.commit()

        fieldnames = list(records[0].keys())
        mode = "a" if self._csv_initialized or self.csv_path.exists() else "w"
        with self.csv_path.open(mode, newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            writer.writerows(records)
        self._csv_initialized = True

    def fetch_all(self) -> List[Dict[str, object]]:
        cursor = self.connection.execute(
            """
            SELECT
                frame_index,
                timestamp_seconds,
                track_id,
                label,
                movement_score,
                head_centered,
                hand_raised,
                upright_posture,
                face_detected,
                excessive_movement
            FROM frame_results
            ORDER BY frame_index, track_id
            """
        )
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        self.connection.close()
