"""SQLite logger: schema exactness + row roundtrip."""

import sqlite3

from orchestrator.logger import REQUEST_COLS, Logger

SPEC_REQUEST_COLS = [  # log schema, order pinned here so figures can rely on it
    "run_id", "req_id", "node_id", "tier", "action", "rssi_dbm",
    "t_assign", "t_capture_us", "t_edge_us",
    "t_first_frag", "t_last_frag", "n_frags", "n_retx",
    "t_queue_in", "t_queue_out", "t_tail_us",
    "t_total_ms", "status", "pred_class", "bytes_on_air",
]


def test_schema_and_roundtrip(tmp_path):
    db = tmp_path / "run.db"
    lg = Logger(db)
    lg.start_run("r1", "k_deep", "{}", "abc123")
    lg.log_request(req_id=1, node_id=11, tier="A", action="k_deep",
                   rssi_dbm=-62, t_assign=1.0, t_capture_us=1000,
                   t_edge_us=50000, t_first_frag=1.01, t_last_frag=1.02,
                   n_frags=1, n_retx=0, t_queue_in=1.02, t_queue_out=1.03,
                   t_tail_us=4000, t_total_ms=35.5, status="OK",
                   pred_class=1, bytes_on_air=900)
    lg.log_request(req_id=2, node_id=31, tier="C", action="k0",
                   rssi_dbm=-75, t_assign=2.0, status="TIMEOUT")
    lg.log_heartbeat(11, -55, 120000, 40, 0, 0xDEADBEEF)
    lg.close()

    conn = sqlite3.connect(db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(requests)")]
    assert cols == SPEC_REQUEST_COLS
    assert ["run_id"] + REQUEST_COLS == SPEC_REQUEST_COLS

    rows = conn.execute("SELECT run_id, status, action, rssi_dbm FROM requests ORDER BY req_id").fetchall()
    assert rows == [("r1", "OK", "k_deep", -62), ("r1", "TIMEOUT", "k0", -75)]
    assert conn.execute("SELECT node_id, fw_hash FROM heartbeats").fetchone() == (11, "deadbeef")
    assert conn.execute("SELECT policy_name, git_hash FROM runs").fetchone() == ("k_deep", "abc123")
    conn.close()
