# qasm/ — benchmark circuit sources

## qasmbench/

OPENQASM 2.0 files from **QASMBench** (PNNL), retrieved file-by-file from
`github.com/pnnl/QASMBench` at the commit recorded in
`qasmbench/SOURCE_COMMIT.txt`; the upstream `LICENSE` file is included
alongside. Please cite the QASMBench paper when these circuits appear in any
publication artifact.

These files are inputs to `scripts/extract_qasm.py`, which derives the
two-qubit-gate skeletons used by this project (Assumption A3: single-qubit
gates dropped; each two-qubit instruction is one gate; `ccx` expands to its
standard 6-CNOT construction, `cswap` to 8 gates; see DESIGN_DECISIONS D30)
and freezes them as explicit instances under `configs/circuits/qasmbench/`.
A drift-guard test (`tests/unit/test_qasm_skeleton.py`) re-extracts every
file and compares against the committed YAMLs.

`supremacy_n120` is NOT a QASMBench circuit: it is generated (10 layers of
seeded random perfect matchings over 120 qubits, seed 2027; guide §10.1,
D28) by the same script.

Panel composition note (D28): the guide §10.1 wish list named some sizes
that do not exist in QASMBench (qaoa_n14, qft_n50, vqe_n80, qpe_n90,
bv_n100). The panel uses real files at the nearest available sizes instead
(qft_n63, cat_n65, ghz_n78, bv_n70, ising_n98, ...). The VQE family ships
only as multi-MB programs with ~3*10^5 two-qubit gates (vqe_uccsd_n28,
vqe_n24) — orders of magnitude beyond simulator-panel scale — and is
replaced by adder_n28 / multiplier_n45 / dnn_n51.
