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


class TierCfg(Strict):
    target: str
    psram: bool
    arena_kb: int
    cuts: list[str]


class ProtocolCfg(Strict):
    assign_ack_timeout_ms: int
    assign_max_retries: int
    nack_delay_ms: int
    max_nack_rounds: int
    t_max_ms: int


class RegistryCfg(Strict):
    heartbeat_period_s: int
    lost_after_s: int


class ExperimentCfg(Strict):
    sweep_reqs_per_cut: int
    min_ok_per_node: int
    node_inflight_cap: int
    request_gap_ms: int


class LoggingCfg(Strict):
    db_dir: str


class SimCfg(Strict):
    goodput_bytes_per_s: dict[str, int]
    rssi_dbm: dict[str, int]
    capture_ms: int
    jpeg_read_ms: int
    t_edge_ms: dict[str, dict[str, int]]
    tail_extra_ms: dict[str, float]


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
    protocol: ProtocolCfg
    registry: RegistryCfg
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

    def cuts_for_tier(self, tier: str) -> list[str]:
        return list(self.tiers[tier].cuts)


def load_config(path: pathlib.Path | None = None) -> Config:
    with open(path or ROOT / "config.yaml") as f:
        return Config(**yaml.safe_load(f))
