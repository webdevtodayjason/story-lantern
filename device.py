#!/usr/bin/env python3
"""
STORY LANTERN - finding the Tiiny.

Firmware 1.0 changed how a box is addressed, twice over, and both halves used to
be fatal here.

The AI gateway no longer answers on port 8800 from another machine. It binds
172.17.0.1:8800, the container bridge only, and every service arrives on port 80
instead, where a router picks one out of the Host header. Older firmware still
serves the gateway on 8800. So the port is a question, not a constant.

And a box's LAN address is a DHCP lease, so it moves. Requiring somebody to type
TIINY_HOST meant the app stopped working the next time the lease turned over, and
it meant an app the farm planted did not start at all, because the farm writes
the address somewhere else.

So nothing here is configured. The address is resolved once, in this order, and
the answer records where it came from so a log can never be ambiguous about which
box produced a story:

    TIINY_BASE                  what the farm CLI exports
    ~/.tiinyapps/device.json    what `farm device` writes: {"base", "key"}
    TIINY_HOST                  what this app documented first. Still honoured
    a scan                      every USB /30 peer, then this host's own /24

The scan is the only step that identifies a box rather than merely finding an
open port, because :39218/device.json is unauthenticated and carries the serial
number. That is what lets a box answering on both Wi-Fi and USB be recognised as
one box, and what makes it safe to prefer the USB address: a /30 handed out by
the cable cannot move, and a DHCP lease can.

Standard library only, like everything else here.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

# Unauthenticated device metadata, and the only endpoint that carries the serial.
DISCO_PORT = 39218

# What the farm hands an app when it gives it a box.
FARM_DEVICE = os.path.join(os.path.expanduser("~"), ".tiinyapps", "device.json")

# A USB-attached Tiiny is a point-to-point /30 inside 172.17/16.
USB_NET = "172.17."

# Names the TiinyOS desktop app writes into /etc/resolver, pointed at a proxy on
# the machine running it. They work there and nowhere else, so they are a last
# resort after the scan finds nothing, not the primary route.
PROXY_HOSTS = ("tiiny", "openai.api.tiiny", "tiiny.local")


@dataclass(frozen=True)
class Device:
    """Where the box is, how we know, and how to talk to it."""
    host: str
    port: int
    key: str
    source: str
    plane: str          # "usb", "lan", "proxy" or "given"
    serial: str = ""

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def describe(self) -> str:
        bits = [self.base_url, f"found by {self.source}", f"{self.plane} plane"]
        if self.serial:
            bits.append(f"serial {self.serial}")
        return "  ".join(bits)


# ---------------------------------------------------------------------------
# where an address is written down
# ---------------------------------------------------------------------------

def farm_device() -> dict:
    """What `farm device` wrote, or an empty dict."""
    try:
        with open(FARM_DEVICE, "rb") as fh:
            got = json.load(fh)
        return got if isinstance(got, dict) else {}
    except Exception:      # absent or unreadable is just "nothing there"
        return {}


def split_base(base) -> tuple:
    """A base URL or bare address as (host, explicit port or None).

    Accepts everything the farm and the environment actually contain:
    "1.2.3.4", "http://1.2.3.4", "http://1.2.3.4:8800/v1", a hostname, or a
    hostname and a port.
    """
    base = (base or "").strip()
    if not base:
        return None, None
    if "//" not in base:
        base = "http://" + base
    try:
        parts = urllib.parse.urlsplit(base)
        return (parts.hostname or None), parts.port
    except ValueError:
        return None, None


def key_from_env() -> str:
    """The bearer key, from the environment or from the farm's device file."""
    return (os.environ.get("TIINY_KEY") or farm_device().get("key") or "").strip()


# ---------------------------------------------------------------------------
# the transport
# ---------------------------------------------------------------------------

def gateway_port(host: str, timeout: float = 2.0) -> int:
    """Which port serves the AI gateway on this box.

    Port 80 answers on every firmware, so a plain TCP probe cannot tell the two
    apart. Asking for an AI route can: the firmware that does not serve it 404s,
    and a 401 still means the gateway is here and wants a key.

    TIINY_PORT pins it, for anyone who has moved it.
    """
    env = os.environ.get("TIINY_PORT")
    if env and env.isdigit():
        return int(env)
    for port in (80, 8800):
        try:
            req = urllib.request.Request(
                "http://%s:%d/v1/models" % (host, port),
                headers={"Authorization": "Bearer probe"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 404:
                    return port
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                return port
        except Exception:          # unreachable on this port; try the next
            continue
    return 80


def reachable(host: str, timeout: float = 2.0) -> bool:
    """Does an AI gateway answer at this address at all?"""
    try:
        req = urllib.request.Request(
            "http://%s:%d/v1/models" % (host, gateway_port(host, timeout)),
            headers={"Authorization": "Bearer probe"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status != 404
    except urllib.error.HTTPError as exc:
        return exc.code != 404
    except Exception:
        return False


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def device_json(addr: str, timeout: float = 0.6):
    """What the box at this address says about itself, or None.

    Unauthenticated, and the reply carries serial_number, which is what lets two
    addresses for one box be recognised as one box.
    """
    try:
        with urllib.request.urlopen(
                "http://%s:%d/device.json" % (addr, DISCO_PORT),
                timeout=timeout) as r:
            got = json.load(r)
    except Exception:
        return None
    if not isinstance(got, dict) or not got.get("serial_number"):
        return None
    return got


def _mask_bits(mask: str) -> int:
    """Prefix length from either form of netmask a system tool prints."""
    if mask.startswith("0x"):
        return bin(int(mask, 16)).count("1")
    parts = [int(x) for x in mask.split(".")]
    if len(parts) != 4:
        raise ValueError(mask)
    n = 0
    for part in parts:
        n = (n << 8) | part
    return bin(n).count("1")


def interfaces() -> list:
    """(address, prefix length) for every IPv4 this machine holds.

    Standard library only, so this asks the system's own tool: `ip` on Linux,
    `ifconfig` on macOS and the BSDs. A machine where neither runs simply
    contributes no candidates.
    """
    out = []
    try:
        txt = subprocess.run(["ip", "-o", "-4", "addr", "show"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in txt.splitlines():
            for field in line.split():
                if "/" in field and field[0].isdigit():
                    addr, _, prefix = field.partition("/")
                    try:
                        out.append((addr, int(prefix)))
                    except ValueError:
                        pass
                    break
    except Exception:
        pass
    if out:
        return out
    try:
        txt = subprocess.run(["ifconfig", "-a"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return out
    for line in txt.splitlines():
        field = line.split()
        if not field or field[0] != "inet" or "netmask" not in field:
            continue
        try:
            out.append((field[1], _mask_bits(field[field.index("netmask") + 1])))
        except (ValueError, IndexError):
            continue
    return out


def usb_peers() -> list:
    """The device end of every attached USB link.

    A Tiiny's USB interface is a point-to-point /30: four addresses, of which the
    box takes the first usable one and this machine the second. So the peer is
    arithmetic rather than a guess, and somebody with several boxes plugged in
    has one of these per cable.
    """
    peers = []
    for addr, bits in interfaces():
        if bits != 30 or not addr.startswith(USB_NET):
            continue
        try:
            octets = [int(x) for x in addr.split(".")]
        except ValueError:
            continue
        n = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
        base = n & ~3
        for cand in (base + 1, base + 2):
            if cand != n:
                peers.append("%d.%d.%d.%d" % (
                    (cand >> 24) & 255, (cand >> 16) & 255,
                    (cand >> 8) & 255, cand & 255))
    return peers


def lan_candidates() -> list:
    """Every other address in the /24 around each of this machine's addresses.

    A /24 is 254 probes, about a second threaded, and it is the only thing that
    finds a box whose DHCP lease moved. Only the /24 this machine sits in:
    sweeping a /16 to locate a Tiiny is not something a bedside appliance should
    do to somebody's network.
    """
    out, seen = [], set()
    for addr, bits in interfaces():
        if addr.startswith("127.") or addr.startswith(USB_NET) or bits >= 31:
            continue
        head = addr.rsplit(".", 1)[0]
        for i in range(1, 255):
            cand = "%s.%d" % (head, i)
            if cand != addr and cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def scan(timeout: float = 0.6, workers: int = 64) -> list:
    """Every Tiiny this machine can see, deduped by serial, USB first.

    A box on Wi-Fi and USB at once answers on both, and the two addresses are one
    device. USB wins the tie, because a /30 handed out by the cable cannot move.
    """
    found, lock = {}, threading.Lock()

    def probe(addr, plane):
        got = device_json(addr, timeout)
        if not got:
            return
        rec = {"addr": addr, "plane": plane,
               "serial": got.get("serial_number"),
               "name": got.get("device_name")}
        with lock:
            cur = found.get(rec["serial"])
            if cur is None or (cur["plane"] == "lan" and plane == "usb"):
                found[rec["serial"]] = rec

    for plane, addrs in (("usb", usb_peers()), ("lan", lan_candidates())):
        for i in range(0, len(addrs), workers):
            threads = [threading.Thread(target=probe, args=(addr, plane))
                       for addr in addrs[i:i + workers]]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    return sorted(found.values(), key=lambda r: (r["plane"] != "usb", r["addr"]))


# ---------------------------------------------------------------------------
# the resolver
# ---------------------------------------------------------------------------

class NotFound(RuntimeError):
    """No Tiiny could be found, with the thing to do about it in the message."""


_CURRENT = None
_LOCK = threading.Lock()


def resolve(host: str = "", serial: str = "") -> Device:
    """Work out which box we are talking to and how. Raises NotFound.

    Does no scanning at all when the address is already written down, which is
    the normal case, so the cost is one gateway probe.
    """
    key = key_from_env()

    def settle(addr, port, source, plane, disco=None):
        if disco is None and plane != "proxy":
            disco = device_json(addr, timeout=1.5) or {}
        return Device(host=addr, port=port or gateway_port(addr), key=key,
                      source=source, plane=plane,
                      serial=(disco or {}).get("serial_number") or "")

    for value, source in ((os.environ.get("TIINY_BASE"), "TIINY_BASE"),
                          (farm_device().get("base"), FARM_DEVICE),
                          (host, "the --host argument"),
                          (os.environ.get("TIINY_HOST"), "TIINY_HOST")):
        addr, port = split_base(value)
        if addr:
            return settle(addr, port, source, "given")

    boxes = scan()
    if serial:
        boxes = [b for b in boxes
                 if b["serial"] == serial or b["serial"].endswith(serial)]
        if not boxes:
            raise NotFound("No Tiiny with serial %r answered." % serial)
    if len(boxes) > 1:
        lines = ["More than one Tiiny answered. Set TIINY_BASE to the one you "
                 "want:", ""]
        for b in boxes:
            lines.append("    %-16s %-4s %-24s %s" % (
                b["addr"], b["plane"], b["serial"], b["name"] or ""))
        raise NotFound("\n".join(lines))
    if boxes:
        b = boxes[0]
        return settle(b["addr"], None, "a scan", b["plane"],
                      {"serial_number": b["serial"]})

    for name in PROXY_HOSTS:
        if reachable(name, timeout=1.5):
            return settle(name, None, "a TiinyOS proxy name", "proxy", {})

    raise NotFound(
        "No Tiiny found. Looked for a USB /30 peer, swept this machine's own "
        "/24 on :%d, and tried %s.\n"
        "Set TIINY_BASE to the device's address, or plug it in."
        % (DISCO_PORT, ", ".join(PROXY_HOSTS)))


def current(host: str = "", serial: str = "") -> Device:
    """The resolved device, worked out once and remembered.

    Deliberately lazy. Resolving at import time meant a module that only wanted
    to define a few functions spent two network timeouts doing it, and made
    `import safety` fail on a machine with no Tiiny on the network, which is
    where the tests run.
    """
    global _CURRENT
    with _LOCK:
        if _CURRENT is None:
            _CURRENT = resolve(host, serial)
        return _CURRENT


def set_current(dev) -> None:
    """Pin the device, for tests and for a caller that has already resolved."""
    global _CURRENT
    with _LOCK:
        _CURRENT = dev


def forget() -> None:
    """Resolve again on the next call, after a lease change or a replug."""
    set_current(None)


if __name__ == "__main__":
    try:
        print("  " + current().describe())
    except NotFound as exc:
        raise SystemExit("  " + str(exc))
