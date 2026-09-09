"""Render the host soft-AP configs (hostapd + dnsmasq with MAC-pinned DHCP reservations)"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator.config import load_config  # noqa: E402

HOSTAPD = """\
interface={iface}
driver=nl80211
ssid={ssid}
hw_mode=g
channel={channel}
ieee80211n=1
wmm_enabled=1
auth_algs=1
wpa=2
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
wpa_passphrase={psk}
"""

DNSMASQ = """\
interface={iface}
bind-interfaces
port=0
dhcp-authoritative
dhcp-range=192.168.4.10,192.168.4.200,255.255.255.0,24h
{reservations}
"""


def main():
    cfg = load_config()
    out = pathlib.Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    placeholders = [n.node_id for n in cfg.nodes if n.mac.startswith("00:00:00")]
    if placeholders:
        print(f"WARNING: nodes {placeholders} still have placeholder MACs in "
              "config.yaml - fill in the real board MACs (printed on boot or "
              "`esptool.py read_mac`) and re-run.")
    res = "\n".join(f"dhcp-host={n.mac},{n.ip},node{n.node_id}" for n in cfg.nodes)
    (out / "hostapd.conf").write_text(HOSTAPD.format(
        iface=cfg.network.ap_interface, ssid=cfg.network.ssid,
        channel=cfg.network.wifi_channel, psk=cfg.network.wpa2_psk))
    (out / "dnsmasq.conf").write_text(DNSMASQ.format(
        iface=cfg.network.ap_interface, reservations=res))
    print(f"wrote {out}/hostapd.conf and {out}/dnsmasq.conf")


if __name__ == "__main__":
    main()
