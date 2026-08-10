# rl-optimizer

> **⚠️ SUPERSEDED — this repository is legacy and no longer maintained (last substantive commit: 2025-01-13).**
>
> Its responsibilities have moved to actively maintained repositories:
>
> - **Trading environments** → [harveybc/gym-fx](https://github.com/harveybc/gym-fx)
> - **Optimization (distributed/evolutionary)** → [harveybc/doin-node](https://github.com/harveybc/doin-node) and [harveybc/agent-multi](https://github.com/harveybc/agent-multi)
>
> Use those repositories for any new work. This repository is retained for historical reference only and is not required by any current deployment.

## What this was

A plugin-based command-line tool for optimizing reinforcement-learning trading agents. It dynamically loaded optimizer, environment, and agent plugins through setuptools entry points (`rl_optimizer.optimizers`, `rl_optimizer.environments`, `rl_optimizer.agents`) and drove an optimize/evaluate loop over CSV time-series data. Registered plugins included an OpenRL optimizer, a PPO agent, a dummy automation agent, and prediction/custom environment wrappers (see [`setup.py`](setup.py) and [`app/plugins/`](app/plugins/)).

## Known defects (why it does not run as documented)

- **Broken entry points:** `setup.py` registers `neat` and `neat_p2p` optimizer plugins pointing at `app.plugins.optimizer_plugin_neat` and `app.plugins.optimizer_plugin_neat_p2p`, but those modules do not exist in [`app/plugins/`](app/plugins/). Selecting either optimizer fails at load time.
- **Broken default configuration:** [`app/config.py`](app/config.py) defaults `optimizer_plugin` to `neat_a_cs`, which is not registered as an entry point at all, so a bare run cannot load its optimizer.
- **Dead dependencies:** [`requirements.txt`](requirements.txt) pins `tensorflow-gpu` (removed from PyPI as a separate package) and `gym` (long deprecated in favor of Gymnasium). A clean install from `requirements.txt` is not expected to succeed on current toolchains.

## Historical usage — unverified in current environments

The commands below reflect how the tool was originally used. They have **not** been re-verified and are expected to fail with current Python/package ecosystems (see defects above).

```bash
git clone https://github.com/harveybc/rl-optimizer.git
cd rl-optimizer
pip install -r requirements.txt   # historical — unverified in current environments
pip install .
rl_optimizer --load_config input_config.json   # historical — unverified in current environments
```

## Limitations

- No maintenance, no issue support, no compatibility work is planned.
- The only optimizer plugin whose module actually exists is the OpenRL one; NEAT-based optimization never shipped in this repository and lives on in the successor stack.
- Test suites and helper scripts (`*.bat`, `*.sh`) target the original 2024-era environment and are unverified.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
