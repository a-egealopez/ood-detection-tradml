from ingestion.event_store import SQLiteDataManager

BATCH = [
    {
        "house_id": "aruba",
        "timestamp": "2024-01-01 10:30:00",
        "sensor_id": "M001",
        "event_type": "motion",
        "value": 1.0,
    },
    {
        "house_id": "aruba",
        "timestamp": "2024-01-01 10:31:00",
        "sensor_id": "M001",
        "event_type": "motion",
        "value": 0.0,
    },
]


def test_duplicate_insert_is_idempotent(tmp_path):
    db = SQLiteDataManager(str(tmp_path / "test.db"))
    db.connect()
    db.create_tables()
    count1 = db.insert_batch(BATCH)
    count2 = db.insert_batch(BATCH)
    assert count1 == 2
    assert count2 == 0
    stats = db.get_stats("aruba")
    assert stats["count"] == 2
    db.close()


def test_query_house_roundtrip(tmp_path):
    db = SQLiteDataManager(str(tmp_path / "roundtrip.db"))
    db.connect()
    db.create_tables()
    db.insert_batch(BATCH)
    df = db.query_house("aruba")
    assert len(df) == 2
    assert list(df["house_id"].unique()) == ["aruba"]
    db.close()


def test_list_houses(tmp_path):
    db = SQLiteDataManager(str(tmp_path / "houses.db"))
    db.connect()
    db.create_tables()
    db.insert_batch(BATCH)
    assert db.list_houses() == ["aruba"]
    db.close()
