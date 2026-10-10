# Tests

Pytest suite. `testpaths = ["tests"]` and `addopts = "-v"` are set in `pyproject.toml`, so a
bare `pytest` from the repository root runs everything here.

```bash
pytest
pytest tests/test_grn.py -k identity
```

`conftest.py` is currently empty; fixtures live next to the tests that use them.

## `test_grn.py`

22 tests for [`../torch_omni_tools`](../torch_omni_tools/blocks/ConvNeXtV2/README.md).
Inputs are generated with a fixed-seed `torch.Generator` so runs are reproducible, and the
block is cross-checked against a straightforward re-implementation of the GRN formula
(`_reference_grn`) rather than against hardcoded numbers.

What is covered:

- **Shapes** — output matches the input for `[1, 3, 1, 1]`, `[2, 8, 5, 7]` and
  `[4, 16, 32, 32]`; non-contiguous inputs; float32/float64 dtype preservation; a channel
  count that does not match `dim` and a non-4D input both fail.
- **Formula** — equality with the reference implementation, norms computed over `H, W`
  before being normalized across `C`, `gamma` scaling and `beta` shifting per channel,
  `beta` acting as a pure additive shift, positive homogeneity when `beta == 0`.
- **Initialization and parameters** — `gamma`/`beta` shapes, both zero-initialized, block is
  the identity at init, `state_dict` round-trip, all-zero input does not produce `NaN`.
- **Autograd** — gradients reach the input and both parameters, input and parameter
  gradients match an autograd reference, `beta` receives a constant gradient, and train and
  eval mode agree.

## Adding tests

Keep tests deterministic (seed the generator), assert against a reference implementation
where one is cheap to write, and prefer `torch.testing.assert_close` over manual
`tolerances`. Type hints are required — mypy runs in strict mode and CI runs both tools.
