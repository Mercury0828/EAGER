"""Phase 0: config loader validation — happy paths and helpful errors."""

import pytest

from eager.config import (
    ConfigError,
    load_circuit_config,
    load_hardware_config,
)


def hw(**over):
    base = {
        "name": "t", "qpus": 2, "topology": "line", "kappa": 4,
        "link_defaults": {"p": 0.5, "W": 2, "B": 8, "T_cut": 20, "w": 1.0},
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- hardware --

def test_shipped_hardware_configs_load(hardware_dir):
    for fname in ("k2_line.yaml", "k4_grid.yaml", "golden_k2_det.yaml"):
        cfg = load_hardware_config(hardware_dir / fname)
        assert cfg.num_qpus >= 2
        assert cfg.num_links >= 1
        assert all(0 < l.p <= 1 for l in cfg.links)


def test_line_topology_edges():
    cfg = load_hardware_config(hw(qpus=4))
    assert [(l.u, l.v) for l in cfg.links] == [(0, 1), (1, 2), (2, 3)]


def test_ring_topology_edges_and_k2_rejected():
    cfg = load_hardware_config(hw(qpus=4, topology="ring"))
    assert [(l.u, l.v) for l in cfg.links] == [(0, 1), (0, 3), (1, 2), (2, 3)]
    with pytest.raises(ConfigError, match="ring"):
        load_hardware_config(hw(qpus=2, topology="ring"))


def test_grid_topology_edges_and_dim_mismatch():
    cfg = load_hardware_config(hw(qpus=4, topology="grid", grid_dims=[2, 2]))
    assert [(l.u, l.v) for l in cfg.links] == [(0, 1), (0, 2), (1, 3), (2, 3)]
    with pytest.raises(ConfigError, match="grid_dims"):
        load_hardware_config(hw(qpus=6, topology="grid", grid_dims=[2, 2]))
    with pytest.raises(ConfigError, match="grid_dims"):
        load_hardware_config(hw(qpus=4, topology="grid"))


def test_explicit_topology_validation():
    cfg = load_hardware_config(
        hw(qpus=3, topology="explicit", edges=[[0, 1], [1, 2]]))
    assert [(l.u, l.v) for l in cfg.links] == [(0, 1), (1, 2)]
    with pytest.raises(ConfigError, match="self-loop"):
        load_hardware_config(hw(qpus=3, topology="explicit", edges=[[0, 0]]))
    with pytest.raises(ConfigError, match="duplicate"):
        load_hardware_config(
            hw(qpus=3, topology="explicit", edges=[[0, 1], [1, 0], [1, 2]]))
    with pytest.raises(ConfigError, match="outside"):
        load_hardware_config(hw(qpus=3, topology="explicit", edges=[[0, 5]]))
    with pytest.raises(ConfigError, match="not connected"):
        load_hardware_config(
            hw(qpus=4, topology="explicit", edges=[[0, 1], [2, 3]]))


def test_unknown_keys_rejected():
    with pytest.raises(ConfigError, match="unknown top-level keys"):
        load_hardware_config(hw(bandwith=3))
    with pytest.raises(ConfigError, match="grid_dims"):
        load_hardware_config(hw(grid_dims=[1, 2]))  # only valid for grid


def test_link_field_validation():
    for bad_p in (0, -0.1, 1.5, "x"):
        with pytest.raises(ConfigError, match="'p'"):
            load_hardware_config(hw(link_defaults={"p": bad_p}))
    for field in ("W", "B"):
        with pytest.raises(ConfigError, match=f"'{field}'"):
            load_hardware_config(hw(link_defaults={field: 0}))
    with pytest.raises(ConfigError, match="T_cut"):
        load_hardware_config(hw(link_defaults={"T_cut": -3}))
    with pytest.raises(ConfigError, match="'w'"):
        load_hardware_config(hw(link_defaults={"w": 0}))
    with pytest.raises(ConfigError, match="unknown link field"):
        load_hardware_config(hw(link_defaults={"q": 1}))


def test_t_cut_null_means_no_cutoff():
    cfg = load_hardware_config(hw(link_defaults={"T_cut": None}))
    assert cfg.links[0].T_cut is None


def test_kappa_forms():
    assert load_hardware_config(hw(kappa=5)).kappa == (5, 5)
    assert load_hardware_config(hw(kappa=[3, 7])).kappa == (3, 7)
    with pytest.raises(ConfigError, match="kappa"):
        load_hardware_config(hw(kappa=[3]))
    with pytest.raises(ConfigError, match=r"kappa\[1\]"):
        load_hardware_config(hw(kappa=[3, 0]))


def test_link_overrides():
    cfg = load_hardware_config(
        hw(qpus=3, link_overrides={"1-2": {"p": 0.9, "W": 4}}))
    assert cfg.links[0].p == 0.5 and cfg.links[1].p == 0.9
    assert cfg.links[1].W == 4 and cfg.links[1].B == 8
    with pytest.raises(ConfigError, match="does not match any edge"):
        load_hardware_config(hw(link_overrides={"0-2": {"p": 0.9}}))
    with pytest.raises(ConfigError, match="'u-v'"):
        load_hardware_config(hw(link_overrides={"zero-one": {"p": 0.9}}))


def test_mode_and_t_ep():
    cfg = load_hardware_config(hw(mode="deterministic", t_ep=3))
    assert cfg.deterministic and cfg.t_ep == 3
    with pytest.raises(ConfigError, match="mode"):
        load_hardware_config(hw(mode="quantum"))
    with pytest.raises(ConfigError, match="t_ep"):
        load_hardware_config(hw(t_ep=0))


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_hardware_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_hardware_config(bad)


def test_link_id_lookup():
    cfg = load_hardware_config(hw(qpus=3))
    assert cfg.link_id(1, 0) == 0 and cfg.link_id(1, 2) == 1
    with pytest.raises(KeyError):
        cfg.link_id(0, 2)


# ----------------------------------------------------------------- circuit --

def test_shipped_circuit_configs_load(circuits_dir):
    g1 = load_circuit_config(circuits_dir / "golden_micro_1.yaml")
    assert g1.kind == "explicit" and g1.num_qubits == 3 and len(g1.gates) == 3
    g2 = load_circuit_config(circuits_dir / "golden_micro_2.yaml")
    assert g2.num_qubits == 4 and len(g2.gates) == 4
    syn = load_circuit_config(circuits_dir / "synthetic_n20_d3.yaml")
    assert syn.kind == "synthetic"
    assert syn.params.num_qubits == 20 and syn.params.num_gates == 60


def test_explicit_circuit_validation():
    base = {"name": "c", "kind": "explicit", "num_qubits": 3,
            "gates": [[0, 1], [1, 2]]}
    assert load_circuit_config(base).gates == ((0, 1), (1, 2))
    with pytest.raises(ConfigError, match="distinct"):
        load_circuit_config({**base, "gates": [[1, 1]]})
    with pytest.raises(ConfigError, match="outside"):
        load_circuit_config({**base, "gates": [[0, 3]]})
    with pytest.raises(ConfigError, match="non-empty"):
        load_circuit_config({**base, "gates": []})
    with pytest.raises(ConfigError, match="num_qubits"):
        load_circuit_config({**base, "num_qubits": 1})
    with pytest.raises(ConfigError, match="'params'"):
        load_circuit_config({**base, "params": {}})


def test_synthetic_circuit_validation():
    base = {"name": "s", "kind": "synthetic",
            "params": {"num_qubits": 10, "density": 3, "seed": 1}}
    cfg = load_circuit_config(base)
    assert cfg.params.num_gates == 30 and cfg.params.seed == 1
    cfg2 = load_circuit_config(
        {"name": "s", "kind": "synthetic",
         "params": {"num_qubits": 10, "num_gates": 17}})
    assert cfg2.params.num_gates == 17 and cfg2.params.seed is None
    with pytest.raises(ConfigError, match="exactly one of"):
        load_circuit_config({"name": "s", "kind": "synthetic",
                             "params": {"num_qubits": 10}})
    with pytest.raises(ConfigError, match="exactly one of"):
        load_circuit_config(
            {"name": "s", "kind": "synthetic",
             "params": {"num_qubits": 10, "density": 3, "num_gates": 5}})
    with pytest.raises(ConfigError, match="unknown params keys"):
        load_circuit_config({"name": "s", "kind": "synthetic",
                             "params": {"num_qubits": 10, "density": 3, "M": 5}})
    with pytest.raises(ConfigError, match="kind"):
        load_circuit_config({"name": "s", "kind": "random"})
    with pytest.raises(ConfigError, match="only valid for kind 'explicit'"):
        load_circuit_config({"name": "s", "kind": "synthetic", "num_qubits": 5,
                             "params": {"num_qubits": 10, "density": 3}})
