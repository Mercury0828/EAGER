"""Configuration schemas and loaders for EAGER.

Two config families (guide §4, §10.2):

- Hardware: QPU graph ``G_P = (U, E_P)`` with per-QPU capacity ``kappa`` and
  per-link parameters ``p`` (per-slot Bernoulli generation success), ``W``
  (parallel generation channels), ``B`` (buffer capacity), ``T_cut``
  (decoherence cutoff; ``null`` = no cutoff), ``w`` (cost per pair), plus the
  simulation ``mode`` (``stochastic`` | ``deterministic``) and ``t_ep`` (fixed
  generation duration in slots, used only in deterministic mode, guide §5.2).

- Circuit: either an explicit two-qubit gate list (interpreted as a temporal
  order; the dependency DAG is derived by per-qubit serialization, guide §4.1)
  or synthetic-generator parameters (guide §10.1).

Loaders accept a YAML file path or an already-parsed dict, validate every
field, and raise :class:`ConfigError` with a precise, actionable message.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised for any malformed or inconsistent configuration."""


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------

LINK_FIELDS = ("p", "W", "B", "T_cut", "w")
TOPOLOGIES = ("line", "ring", "grid", "explicit")
MODES = ("stochastic", "deterministic")

_HW_KEYS = {
    "name", "qpus", "topology", "grid_dims", "edges", "kappa",
    "link_defaults", "link_overrides", "mode", "t_ep",
}
_LINK_DEFAULTS = {"p": 1.0 / 12.0, "W": 2, "B": 8, "T_cut": 20, "w": 1.0}


@dataclass(frozen=True)
class LinkConfig:
    """One undirected link (u, v) with u < v. ``T_cut is None`` means no cutoff."""

    u: int
    v: int
    p: float
    W: int
    B: int
    T_cut: int | None
    w: float

    @property
    def endpoints(self) -> tuple[int, int]:
        return (self.u, self.v)


@dataclass(frozen=True)
class HardwareConfig:
    name: str
    num_qpus: int
    topology: str
    kappa: tuple[int, ...]
    links: tuple[LinkConfig, ...]          # index in this tuple == link id
    mode: str                              # "stochastic" | "deterministic"
    t_ep: int                              # deterministic generation duration

    @property
    def num_links(self) -> int:
        return len(self.links)

    @property
    def deterministic(self) -> bool:
        return self.mode == "deterministic"

    def link_id(self, u: int, v: int) -> int:
        a, b = (u, v) if u < v else (v, u)
        for i, l in enumerate(self.links):
            if (l.u, l.v) == (a, b):
                return i
        raise KeyError(f"no link between QPU {u} and QPU {v}")

    def summary(self) -> str:
        lines = [
            f"hardware '{self.name}': K={self.num_qpus} topology={self.topology} "
            f"mode={self.mode} t_ep={self.t_ep}",
            f"  kappa={list(self.kappa)} (total {sum(self.kappa)})",
        ]
        for i, l in enumerate(self.links):
            cut = "inf" if l.T_cut is None else l.T_cut
            lines.append(
                f"  link {i}: ({l.u},{l.v}) p={l.p:.6g} W={l.W} B={l.B} "
                f"T_cut={cut} w={l.w:.6g}"
            )
        return "\n".join(lines)


def _ctx(name: str, msg: str) -> ConfigError:
    return ConfigError(f"hardware config '{name}': {msg}")


def _require_int(name: str, field: str, val: Any, lo: int = 1) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise _ctx(name, f"field '{field}' must be an integer, got {val!r}")
    if val < lo:
        raise _ctx(name, f"field '{field}' must be >= {lo}, got {val}")
    return val


def _build_edges(name: str, topology: str, k: int, raw: Mapping[str, Any]
                 ) -> list[tuple[int, int]]:
    if topology == "line":
        if k < 2:
            raise _ctx(name, f"topology 'line' needs qpus >= 2, got {k}")
        return [(i, i + 1) for i in range(k - 1)]
    if topology == "ring":
        if k < 3:
            raise _ctx(name, f"topology 'ring' needs qpus >= 3, got {k} "
                             "(a 2-QPU ring duplicates the line edge; use 'line')")
        return [(i, i + 1) for i in range(k - 1)] + [(0, k - 1)]
    if topology == "grid":
        dims = raw.get("grid_dims")
        if (not isinstance(dims, (list, tuple)) or len(dims) != 2
                or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 1
                           for d in dims)):
            raise _ctx(name, "topology 'grid' requires grid_dims: [rows, cols] "
                             f"with positive integers, got {dims!r}")
        r, c = dims
        if r * c != k:
            raise _ctx(name, f"grid_dims {r}x{c} = {r*c} does not match qpus={k}")
        edges = []
        for i in range(r):
            for j in range(c):
                n = i * c + j
                if j + 1 < c:
                    edges.append((n, n + 1))
                if i + 1 < r:
                    edges.append((n, n + c))
        return edges
    if topology == "explicit":
        raw_edges = raw.get("edges")
        if not isinstance(raw_edges, list) or not raw_edges:
            raise _ctx(name, "topology 'explicit' requires a non-empty 'edges' list")
        seen: set[tuple[int, int]] = set()
        edges = []
        for e in raw_edges:
            if (not isinstance(e, (list, tuple)) or len(e) != 2
                    or not all(isinstance(x, int) and not isinstance(x, bool) for x in e)):
                raise _ctx(name, f"each edge must be a pair of integers, got {e!r}")
            u, v = e
            if u == v:
                raise _ctx(name, f"self-loop edge [{u},{v}] is not allowed")
            if not (0 <= u < k and 0 <= v < k):
                raise _ctx(name, f"edge [{u},{v}] references a QPU outside 0..{k-1}")
            key = (min(u, v), max(u, v))
            if key in seen:
                raise _ctx(name, f"duplicate edge [{key[0]},{key[1]}]")
            seen.add(key)
            edges.append(key)
        return edges
    raise _ctx(name, f"unknown topology {topology!r}; expected one of {TOPOLOGIES}")


def _check_connected(name: str, k: int, edges: list[tuple[int, int]]) -> None:
    adj: dict[int, list[int]] = {i: [] for i in range(k)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for n in adj[cur]:
            if n not in seen:
                seen.add(n)
                stack.append(n)
    if len(seen) != k:
        missing = sorted(set(range(k)) - seen)
        raise _ctx(name, f"QPU graph is not connected; unreachable QPUs: {missing}")


def _validate_link_field(name: str, field: str, val: Any, where: str) -> Any:
    if field == "p":
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise _ctx(name, f"field 'p' must be a number in (0, 1], got {val!r} ({where})")
        if not (0.0 < float(val) <= 1.0):
            raise _ctx(name, f"field 'p' must be in (0, 1], got {val} ({where})")
        return float(val)
    if field in ("W", "B"):
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise _ctx(name, f"field '{field}' must be an integer >= 1, got {val!r} ({where})")
        return val
    if field == "T_cut":
        if val is None:
            return None
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise _ctx(name, f"field 'T_cut' must be an integer >= 1 or null (no cutoff), "
                             f"got {val!r} ({where})")
        return val
    if field == "w":
        if isinstance(val, bool) or not isinstance(val, (int, float)) or float(val) <= 0:
            raise _ctx(name, f"field 'w' must be a number > 0, got {val!r} ({where})")
        return float(val)
    raise _ctx(name, f"unknown link field '{field}' ({where}); expected one of {LINK_FIELDS}")


def load_hardware_config(source: str | Path | Mapping[str, Any]) -> HardwareConfig:
    raw, default_name = _read_source(source, "hardware")
    name = raw.get("name", default_name)
    if not isinstance(name, str) or not name:
        raise ConfigError(f"hardware config: field 'name' must be a non-empty string, got {name!r}")

    unknown = set(raw) - _HW_KEYS
    if unknown:
        raise _ctx(name, f"unknown top-level keys {sorted(unknown)}; "
                         f"expected a subset of {sorted(_HW_KEYS)}")

    if "qpus" not in raw:
        raise _ctx(name, "missing required field 'qpus' (number of QPUs, K)")
    k = _require_int(name, "qpus", raw["qpus"], lo=2)

    topology = raw.get("topology")
    if topology not in TOPOLOGIES:
        raise _ctx(name, f"field 'topology' must be one of {TOPOLOGIES}, got {topology!r}")
    if topology != "grid" and "grid_dims" in raw:
        raise _ctx(name, "'grid_dims' is only valid with topology 'grid'")
    if topology != "explicit" and "edges" in raw:
        raise _ctx(name, "'edges' is only valid with topology 'explicit'")

    edges = _build_edges(name, topology, k, raw)
    edges = sorted(edges)                      # link id = index in sorted edge list
    _check_connected(name, k, edges)

    kappa_raw = raw.get("kappa")
    if kappa_raw is None:
        raise _ctx(name, "missing required field 'kappa' (per-QPU computing-qubit capacity)")
    if isinstance(kappa_raw, int) and not isinstance(kappa_raw, bool):
        if kappa_raw < 1:
            raise _ctx(name, f"field 'kappa' must be >= 1, got {kappa_raw}")
        kappa = tuple([kappa_raw] * k)
    elif isinstance(kappa_raw, list):
        if len(kappa_raw) != k:
            raise _ctx(name, f"field 'kappa' list must have length qpus={k}, "
                             f"got length {len(kappa_raw)}")
        for i, x in enumerate(kappa_raw):
            if isinstance(x, bool) or not isinstance(x, int) or x < 1:
                raise _ctx(name, f"kappa[{i}] must be an integer >= 1, got {x!r}")
        kappa = tuple(kappa_raw)
    else:
        raise _ctx(name, f"field 'kappa' must be an integer or a list of {k} integers, "
                         f"got {kappa_raw!r}")

    defaults = dict(_LINK_DEFAULTS)
    ld = raw.get("link_defaults", {})
    if not isinstance(ld, Mapping):
        raise _ctx(name, f"'link_defaults' must be a mapping, got {ld!r}")
    for key, val in ld.items():
        defaults[key] = _validate_link_field(name, key, val, "link_defaults")

    overrides_raw = raw.get("link_overrides", {})
    if not isinstance(overrides_raw, Mapping):
        raise _ctx(name, f"'link_overrides' must be a mapping keyed 'u-v', got {overrides_raw!r}")
    edge_index = {e: i for i, e in enumerate(edges)}
    per_link: dict[int, dict[str, Any]] = {}
    for key, fields in overrides_raw.items():
        try:
            u_s, v_s = str(key).split("-")
            u, v = int(u_s), int(v_s)
        except ValueError:
            raise _ctx(name, f"link_overrides key {key!r} must look like 'u-v' "
                             "(e.g. '0-1')") from None
        e = (min(u, v), max(u, v))
        if e not in edge_index:
            raise _ctx(name, f"link_overrides key '{key}' does not match any edge; "
                             f"edges are {edges}")
        if not isinstance(fields, Mapping):
            raise _ctx(name, f"link_overrides['{key}'] must be a mapping of link fields")
        ov = {}
        for f, val in fields.items():
            ov[f] = _validate_link_field(name, f, val, f"link_overrides['{key}']")
        per_link[edge_index[e]] = ov

    mode = raw.get("mode", "stochastic")
    if mode not in MODES:
        raise _ctx(name, f"field 'mode' must be one of {MODES}, got {mode!r}")
    t_ep = _require_int(name, "t_ep", raw.get("t_ep", 12), lo=1)

    links = []
    for i, (u, v) in enumerate(edges):
        vals = dict(defaults)
        vals.update(per_link.get(i, {}))
        links.append(LinkConfig(u=u, v=v, p=vals["p"], W=vals["W"], B=vals["B"],
                                T_cut=vals["T_cut"], w=vals["w"]))

    return HardwareConfig(
        name=name, num_qpus=k, topology=topology, kappa=kappa,
        links=tuple(links), mode=mode, t_ep=t_ep,
    )


# --------------------------------------------------------------------------
# Circuit
# --------------------------------------------------------------------------

_CIRC_KEYS = {"name", "kind", "num_qubits", "gates", "params"}
_SYNTH_KEYS = {"num_qubits", "density", "num_gates", "seed"}


@dataclass(frozen=True)
class SynthParams:
    num_qubits: int
    num_gates: int
    seed: int | None


@dataclass(frozen=True)
class CircuitConfig:
    name: str
    kind: str                                   # "explicit" | "synthetic"
    num_qubits: int | None                      # explicit only
    gates: tuple[tuple[int, int], ...] | None   # explicit only
    params: SynthParams | None                  # synthetic only


def _cctx(name: str, msg: str) -> ConfigError:
    return ConfigError(f"circuit config '{name}': {msg}")


def load_circuit_config(source: str | Path | Mapping[str, Any]) -> CircuitConfig:
    raw, default_name = _read_source(source, "circuit")
    name = raw.get("name", default_name)
    if not isinstance(name, str) or not name:
        raise ConfigError(f"circuit config: field 'name' must be a non-empty string, got {name!r}")

    unknown = set(raw) - _CIRC_KEYS
    if unknown:
        raise _cctx(name, f"unknown top-level keys {sorted(unknown)}; "
                          f"expected a subset of {sorted(_CIRC_KEYS)}")

    kind = raw.get("kind")
    if kind not in ("explicit", "synthetic"):
        raise _cctx(name, f"field 'kind' must be 'explicit' or 'synthetic', got {kind!r}")

    if kind == "explicit":
        if "params" in raw:
            raise _cctx(name, "'params' is only valid for kind 'synthetic'")
        n = raw.get("num_qubits")
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise _cctx(name, f"field 'num_qubits' must be an integer >= 2, got {n!r}")
        raw_gates = raw.get("gates")
        if not isinstance(raw_gates, list) or not raw_gates:
            raise _cctx(name, "field 'gates' must be a non-empty list of [q_a, q_b] pairs")
        gates = []
        for i, g in enumerate(raw_gates):
            if (not isinstance(g, (list, tuple)) or len(g) != 2
                    or not all(isinstance(x, int) and not isinstance(x, bool) for x in g)):
                raise _cctx(name, f"gates[{i}] must be a pair of integers, got {g!r}")
            a, b = g
            if a == b:
                raise _cctx(name, f"gates[{i}] operates twice on qubit {a}; "
                                  "two-qubit gates need distinct operands")
            if not (0 <= a < n and 0 <= b < n):
                raise _cctx(name, f"gates[{i}] = [{a},{b}] references a qubit outside 0..{n-1}")
            gates.append((a, b))
        return CircuitConfig(name=name, kind="explicit", num_qubits=n,
                             gates=tuple(gates), params=None)

    # synthetic
    for f in ("num_qubits", "gates"):
        if f in raw:
            raise _cctx(name, f"'{f}' is only valid for kind 'explicit'; "
                              "synthetic instances are described under 'params'")
    params = raw.get("params")
    if not isinstance(params, Mapping):
        raise _cctx(name, "kind 'synthetic' requires a 'params' mapping")
    unknown = set(params) - _SYNTH_KEYS
    if unknown:
        raise _cctx(name, f"unknown params keys {sorted(unknown)}; "
                          f"expected a subset of {sorted(_SYNTH_KEYS)}")
    n = params.get("num_qubits")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise _cctx(name, f"params.num_qubits must be an integer >= 2, got {n!r}")
    has_density = "density" in params
    has_gates = "num_gates" in params
    if has_density == has_gates:
        raise _cctx(name, "params must contain exactly one of 'density' (M/N ratio) "
                          "or 'num_gates'")
    if has_density:
        d = params["density"]
        if isinstance(d, bool) or not isinstance(d, (int, float)) or float(d) <= 0:
            raise _cctx(name, f"params.density must be a number > 0, got {d!r}")
        m = max(1, round(float(d) * n))
    else:
        m = params["num_gates"]
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise _cctx(name, f"params.num_gates must be an integer >= 1, got {m!r}")
    seed = params.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise _cctx(name, f"params.seed must be a non-negative integer or absent, got {seed!r}")
    return CircuitConfig(name=name, kind="synthetic", num_qubits=None, gates=None,
                         params=SynthParams(num_qubits=n, num_gates=m, seed=seed))


# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------

def _read_source(source: str | Path | Mapping[str, Any], family: str
                 ) -> tuple[dict, str]:
    """Return (raw dict, default name). Accepts a YAML path or a dict."""
    if isinstance(source, Mapping):
        return dict(source), f"<inline-{family}>"
    path = Path(source)
    if not path.exists():
        raise ConfigError(f"{family} config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{family} config {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{family} config {path} must contain a YAML mapping, "
                          f"got {type(raw).__name__}")
    return raw, path.stem
