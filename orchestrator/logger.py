"""SQLite logging"""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, started_at TEXT, policy_name TEXT,
  trace_file TEXT, config_json TEXT, git_hash TEXT);
CREATE TABLE IF NOT EXISTS requests(
  run_id TEXT, req_id INTEGER, node_id INTEGER, tier TEXT, phase_id INTEGER,
  action TEXT, ctx_rssi_bin INTEGER, ctx_load INTEGER,
  t_assign REAL, t_capture_us INTEGER, t_edge_us INTEGER,
  t_first_frag REAL, t_last_frag REAL, n_frags INTEGER, n_retx INTEGER,
  t_queue_in REAL, t_queue_out REAL, t_tail_us INTEGER,
  t_total_ms REAL, status TEXT, reward REAL, pred_class INTEGER,
  bytes_on_air INTEGER);
CREATE TABLE IF NOT EXISTS heartbeats(
  run_id TEXT, ts REAL, node_id INTEGER, rssi INTEGER, free_heap INTEGER,
  temp INTEGER, boot_count INTEGER, fw_hash TEXT);
"""

REQUEST_COLS = ("req_id node_id tier phase_id action ctx_rssi_bin ctx_load "
                "t_assign t_capture_us t_edge_us t_first_frag t_last_frag "
                "n_frags n_retx t_queue_in t_queue_out t_tail_us t_total_ms "
                "status reward pred_class bytes_on_air").split()


class Logger:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self.run_id = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.conn = None
        self.executor.submit(self._open).result()

    def _open(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def start_run(self, run_id, policy_name, trace_file, config_json, git_hash):
        self.run_id = run_id
        self.executor.submit(
            self._exec, "INSERT INTO runs VALUES (?,?,?,?,?,?)",
            (run_id, time.strftime("%Y-%m-%dT%H:%M:%S"), policy_name,
             trace_file, config_json, git_hash)).result()

    def _exec(self, sql, params):
        self.conn.execute(sql, params)
        self.conn.commit()

    def log_request(self, **row):
        vals = (self.run_id,) + tuple(row.get(c) for c in REQUEST_COLS)
        sql = f"INSERT INTO requests VALUES ({','.join('?' * (len(REQUEST_COLS) + 1))})"
        self.executor.submit(self._exec, sql, vals)

    def log_heartbeat(self, node_id, rssi, free_heap, temp, boot_count, fw_hash):
        self.executor.submit(
            self._exec, "INSERT INTO heartbeats VALUES (?,?,?,?,?,?,?,?)",
            (self.run_id, time.monotonic(), node_id, rssi, free_heap, temp,
             boot_count, f"{fw_hash:08x}"))

    def close(self):
        def _close():
            self.conn.commit()
            self.conn.close()
        self.executor.submit(_close).result()
        self.executor.shutdown(wait=True)
