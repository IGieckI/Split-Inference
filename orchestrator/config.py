"""Typed view of config.yaml (single source of truth)."""

import pathlib

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = pathlib.Path(__file__).resolve().parents[1]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ports(Strict):
    control: int
    data: int
    timesync: int


class NetworkCfg(Strict):
    ssid: str
    wpa2_psk: str
    subnet: str
    server_ip: str
    wifi_channel: int
    ports: Ports
    fragment_payload_bytes: int


class ModelCfg(Strict):
    input_size: int
    width_multiplier: float
    num_classes: int
    jpeg_quality: int
    calibration_images: int
    int8_accuracy_gate: float
    flash_image_count: int
    artifacts_dir: str


class CutCfg(Strict):
    name: str
    boundary_op: int | None
    tensor_name: str | None


class NodeCfg(Strict):
    node_id: int
    tier: str
    ip: str
    mac: str
    input: str  # flash | camera


class TierCfg(Strict):
    target: str
    psram: bool
    arena_kb: int
    arms: list[str]


class EpsilonCfg(Strict):
    start: float
    decay: float
    floor: float


class PolicyCfg(Strict):
    kind: str
    epsilon: EpsilonCfg
    rssi_bins_dbm: list[int]


class RewardCfg(Strict):
    failure_penalty: float
    warmup_runs: int


class ProtocolCfg(Strict):
    assign_ack_timeout_ms: int
    assign_max_retries: int
    nack_delay_ms: int
    max_nack_rounds: int
    t_max_multiplier: float


class RegistryCfg(Strict):
    heartbeat_period_s: int
    lost_after_s: int


class TimesyncCfg(Strict):
    period_s: int
    fit_window: int


class ExperimentCfg(Strict):
    learning_min_ok_per_node: int
    sweep_reqs_per_arm: int
    fleet_inflight_cap: int
    node_inflight_cap: int
    request_gap_ms: int


class LoggingCfg(Strict):
    db_dir: str


class SimCfg(Strict):
    goodput_bytes_per_s: dict[str, int]
    capture_ms: dict[str, int]
    jpeg_encode_ms: dict[str, int]
    t_edge_ms: dict[str, dict[str, int]]


class ProjectCfg(Strict):
    name: str
    phase: int


class Config(Strict):
    project: ProjectCfg
    network: NetworkCfg
    model: ModelCfg
    cuts: list[CutCfg]
    nodes: list[NodeCfg]
    tiers: dict[str, TierCfg]
    policy: PolicyCfg
    reward: RewardCfg
    protocol: ProtocolCfg
    registry: RegistryCfg
    timesync: TimesyncCfg
    experiment: ExperimentCfg
    logging: LoggingCfg
    sim: SimCfg

    # derived helpers
    def node(self, node_id: int) -> NodeCfg:
        return next(n for n in self.nodes if n.node_id == node_id)

    def cut_names(self) -> list[str]:
        return [c.name for c in self.cuts]

    def cut_index(self, name: str) -> int:
        return self.cut_names().index(name)

    def arms_for_tier(self, tier: str) -> list[str]:
        return list(self.tiers[tier].arms)

    def rssi_bin(self, rssi_dbm: int) -> int:
        """0 = good (>= -55), 1 = mid (>= -70), 2 = bad."""
        good, mid = self.policy.rssi_bins_dbm
        if rssi_dbm >= good:
            return 0
        if rssi_dbm >= mid:
            return 1
        return 2


def load_config(path: pathlib.Path | None = None) -> Config:
    with open(path or ROOT / "config.yaml") as f:
        return Config(**yaml.safe_load(f))
