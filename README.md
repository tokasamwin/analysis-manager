# Analysis Manager

A small python based tool for managing scientific and engineering analysis.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Background

This was a core capability of the Hybrid Advanced Reactor Evaluator ([HARE](https://github.com/tokasamwin/hare)) tool and has been separated to provide a more general capability.

## Install
TODO: publish on PyPI

```
pip install pip@git+https://github.com/tokasamwin/analysis-manager
```



## Usage


The aim of this is to provide a simple interface for turning basic scripts into parametric workflows which intuitively match how engineering and scientific analysis is carried out:

* In stages, with only successful design points proceeding to later stages
* Within a design space represented by parameters and their bounds
* Used for parameter searches, design point optimisation, or surrogate modelling

These aspects are all implemented in a modular way. Get your workflow working with a search over a broad parameter space, narrow your parameter bounds and carry out a Multi-Objective Optimisation to find a Pareto front, or update the workflow to use a more detailed assessment stage. 

### Core concepts

* **Sampler**: defines the parameters to be sampled, including continuous and discrete/categorical parameters. This provides an interface to generate a DataFrame of inputs sampled in different methods (currently only using Grid and Sobol sampling methods)
* **Stages**: scientific and engineering analysis is typically staged, including pre-processing, processing and post-processing stages. Wrap your functions in a stage and optionally provide a gate:
    * These gates could be parameter gates, or a check for whether a processing stage successfully completed, or other types of validation
    * Chain stages together, such as building CAD, conducting a transient thermal assessment, a structural assessment, and finally a creep-fatigue assessment for data points which have acceptable temperatures and maximum stresses
* **Routines**: search across your parameter space, optimise across it, or generate a surrogate model. Optimisation uses wrapped methods from [PyMOO](https://pymoo.org)
* **Run manager**: a central run manager keeps track of your design points and provides the relevant path to store data for each assessment:
    * The run manager can chain different routines, such as a search routine leading to a starting point for an optimisation routine

### Run Manager structure

```
|- Design Point
   |- Stage
      |- Assessment files
      |- Input.json
   |- Run configuration data
```

## Web API & Docker

analysisManager can run as a web service: clients submit **run specs** (JSON) containing samplers, stages and routines, and the server executes them inside its process/container, one run at a time behind a FIFO queue.

```bash
pip install .[server]          # base package + flask + gunicorn
gunicorn analysisManager.server:app --bind 0.0.0.0:8080 --workers 1 --threads 8
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness (queue/running state) |
| `GET /api/config` | data dir, registry, version |
| `POST /api/runs` | submit a run spec → `202 {run_id, ...}` |
| `GET /api/runs` | list runs (live state wins over disk), `?limit=` |
| `GET /api/runs/{id}` | status, per-routine progress, design points completed |
| `GET /api/runs/{id}/results?format=csv\|json` | the accumulated results DataFrame |
| `GET /api/runs/{id}/dashboard` | self-refreshing HTML dashboard |
| `DELETE /api/runs/{id}` | cancel — not implemented yet (`501`) |

### Durable outputs

Each run leaves a durable record in `dataDir/<run name>/` so results survive server restarts:

```
dataDir/<run name>/
├── run.json            run status & metadata (written at queued → running → finished)
│     {run_id, name, status, spec, workers, timestamps, error, traceback,
│      version, routines: [{name, state}]}
├── results.csv         joined results table (identical bytes to the live endpoint)
├── config-<hash>.json  run configuration (library)
└── <ID>/               one folder per design point
    ├── inputs.json
    ├── results.json    written only if all stages passed
    └── <stage>/…       your stage functions' artifacts
```

Status values: `queued`, `running`, `succeeded`, `failed`, `interrupted`. At server start, any run still marked `queued`/`running` on disk is re-marked `interrupted` (its per-design-point artifacts are preserved; the joined table was in memory and is lost).

Because the **run name** is the disk folder name, it is the stable key across restarts — `GET /api/runs/{name}` and `GET /api/runs/{run_id}` both resolve from disk. Fire-and-forget workflow:

```bash
RID=$(curl -s -X POST localhost:8080/api/runs -H 'Content-Type: application/json' \
      -d @spec.json | python -c 'import sys, json; print(json.load(sys.stdin)["run_id"])')
# …minutes later, even if the container restarted in between…
curl -s localhost:8080/api/runs/$RID                 # status
curl -sO  localhost:8080/api/runs/$RID/results       # download results CSV
```

### Run spec

```json
{
  "name": "example",
  "workers": "threads",
  "stages": [
    {"name": "area",
     "inputs": ["depth", "width"],
     "function": {"code": "def f(depth, width, **kwargs):\n\treturn {'area': depth * width}", "function": "f"},
     "gate": {"code": "lambda row: row.get('area', 0) > 5"}}
  ],
  "routines": [
    {"type": "search",
     "sampler": {"kind": "basic", "variables": {"depth": [1, 10], "width": [2, 5]}},
     "density": 5},
    {"type": "optimise",
     "sampler": {"kind": "basic", "variables": {"depth": [1, 10], "width": [2, 5]}},
     "target": {"code": "lambda row: -row.get('volume', 1e6)"},
     "pop_size": 20, "n_gen": 20}
  ]
}
```

See `examples/api_client_example.py` for a ready-made client.

### Stage functions across the wire

Every callable slot (`function`, `gate`, `target`, surrogate `model`) takes one of:

* `{"name": "package.module.function"}` or `{"name": "registry-alias"}` — a real importable function, resolved through the **registry** (a JSON file of aliases → dotted paths, from the `ANALYSIS_STAGE_REGISTRY` env var; see `docker/registry.example.json`).
* `{"code": "lambda row: ...", "function": "optional_name"}` — inline Python, executed in a restricted namespace (`np`, `pd`, `math`, `inf` available; imports and `os`/`subprocess`/`open` and friends are blocked).

**Worker modes** (`"workers"`):

| Mode | Registry functions | Inline code | Notes |
|---|---|---|---|
| `none` (default: `threads`) | ✅ | ✅ | sequential |
| `threads` | ✅ | ✅ | in-process thread pool; best when stages wrap external binaries (the common HPC pattern) |
| `processes` | ✅ | ❌ → `422` | parallel worker processes; inline code cannot be pickled into fresh processes |

### Environment variables

| Variable | Meaning |
|---|---|
| `ANALYSISMANAGER_DATADIR` | data directory for run outputs (no interactive prompt when set) |
| `ANALYSIS_STAGE_REGISTRY` | path to the stage-function registry JSON file |
| `ANALYSIS_WORKERS` | default `workerCount` for `threads`/`processes` modes |

### Docker

```bash
docker compose up --build
```

mounts `./data` to `/data` for results, an optional registry file at `/etc/analysis/registry.json`, and exposes port `8080`. Bake any analysis tools your stage functions need into the image (extend the `builder` stage of `Dockerfile`).

## Contributing

PRs accepted.

## License

[MIT © Sam Ha](../LICENSE)