"""SQLite logger: section 11.7 schema exactness + row roundtrip."""

import sqlite3

from orchestrator.logger import REQUEST_COLS, Logger

SPEC_REQUEST_COLS = [  # log schema, order pinned here so figures can rely on it
    "run_id", "req_id", "node_id", "tier", "action",
    "ctx_rssi_bin", "ctx_load",
    "t_assign", "t_capture_us", "t_edge_us",
    "t_first_frag", "t_last_frag", "n_frags", "n_retx",
    "t_queue_in", "t_queue_out", "t_tail_us",
    "t_total_ms", "status", "reward", "pred_class", "bytes_on_air",
]


def test_schema_and_roundtrip(tmp_path):
    db = tmp_path / "run.db"
    lg = Logger(db)
    lg.start_run("r1", "eps_greedy_s0", "{}", "abc123")
    lg.log_request(req_id=1, node_id=11, tier="A", action="k_deep",
                   ctx_rssi_bin=0, ctx_load=1, t_assign=1.0, t_capture_us=1000,
                   t_edge_us=50000, t_first_frag=1.01, t_last_frag=1.02,
                   n_frags=1, n_retx=0, t_queue_in=1.02, t_queue_out=1.03,
                   t_tail_us=4000, t_total_ms=35.5, status="OK", reward=-0.7,
                   pred_class=1, bytes_on_air=900)
    lg.log_request(req_id=2, node_id=31, tier="C", action="k0",
                   ctx_rssi_bin=2, ctx_load=0, t_assign=2.0, status="TIMEOUT",
                   reward=-3.0)
    lg.log_heartbeat(11, -55, 120000, 40, 0, 0xDEADBEEF)
    lg.close()

    conn = sqlite3.connect(db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(requests)")]
    assert cols == SPEC_REQUEST_COLS
    assert ["run_id"] + REQUEST_COLS == SPEC_REQUEST_COLS

    rows = conn.execute("SELECT run_id, status, reward, action FROM requests ORDER BY req_id").fetchall()
    assert rows == [("r1", "OK", -0.7, "k_deep"), ("r1", "TIMEOUT", -3.0, "k0")]
    assert conn.execute("SELECT node_id, fw_hash FROM heartbeats").fetchone() == (11, "deadbeef")
    assert conn.execute("SELECT policy_name, git_hash FROM runs").fetchone() == ("eps_greedy_s0", "abc123")
    conn.close()
