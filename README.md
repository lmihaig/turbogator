# TurboGator

Implementation and optimization of the [GATr paper](https://arxiv.org/pdf/2305.18415).

### Hardware Details

heads up on the environment our code runs in:

* The container has access to 2 cores, but our code is pinned to always run on **CPU 2**.
* 12 GB RAM allocated.
* Frequency is locked to **800 MHz** with turboboost disabled (i'm still confused about this, we're supposed to disable it right?)
* ISA extensions available: **SSE4.2** and **AVX2**.

For the exact specs, check out the `hardware_specs/` folder.

---

### Setup

I think the only thing you guys need to install locally is [uv](https://docs.astral.sh/uv/getting-started/installation/)

*Optional:* Install a GUI like `kcachegrind` so you can actually read the valgrind output without losing your mind.

---

### Developer Workflow

I tried to automate the annoying parts. Here is the loop for making and testing changes. If you guys want to change anything about this flow, go for it, just give a heads up.

1. **Write Code:** Implement your changes or a new backend inside the `turbogator/` directory.
2. **Local Sanity Check:** Run `./debug_local.sh`. This compiles your code locally and runs the validation pipeline to make sure it produces the exact expected output (checks against `baseline/expected.bin`).
3. **Submit Job:** Once it works locally, run `./submit.sh "Brief description of your run"`. This packages your `turbogator/` folder and `config.py` and sends it to the server for benchmarking.
4. **Get Results:** The script will download the results directly into `results/raw/<timestamp_user>/`. This run will also be automatically appended to your `results/history.jsonl` file.
5. **Plotting:** * If there are garbage/failed runs you don't want to show up on the graph, just comment them out in `results/history.jsonl`.
    * Run `./plot.sh` to generate the updated graphs. (Right now it just plots performance, but we should add roofline and perf counter graphs(?) eventually).
6. **Analyze & Optimize:** Go into `results/raw/<timestamp_user>/` folder and check:
    * `callgrind.out.128` (open with kcachegrind to find hotspots)
    * `llvm_mca.txt` (see static port pressure)
    * `metrics.json` (check cycles, cache misses, and top-down performance counters)
7. **Repeat.**

---

### Repo Architecture

```python
.
├── config.py                 # main config file
├── turbogator/               # CORE WORK DIR
├── baseline/                 # Reference Implementation
│   ├── ezgatr/               # baseline repo 
│   ├── benchmark_pytorch.py  # should not be run usually
│   └── generate_validation_data.py # ground truth .bin files
├── analysis/                 # data post processing
│   ├── plot_performance.py   
│   └── plot_style.py         # reusable wrappers to maintain style
├── results/                  
│   ├── baseline/             # metrics, run logs, and validation .bin files
│   ├── raw/                  # per job results
│   └── <timestamp_user>
│           ├── build.log
│           ├── callgrind.out.128 # callgrind output
│           ├── config.py     # config that generated this run
│           ├── llvm_mca.txt  # mca simulation
│           ├── metrics.json  # main metrics file with cycle and perf counters
│           ├── run.log       
│           └── turbogator    # src code that generated this run
│   ├── plots/                
│   └── history.jsonl         # ledger of all your runs
├── server/                   # server side logic, not directly relevant 
├── hardware_specs/           # container, cpu, micro arch info 
├── README.md                 
├── project_questions.md      # answers to the questions on the project system 
│
# --- Workflow Scripts ---
├── submit.sh                 # packages src/ and submits benchmarking job to server
├── submit_baseline.sh        # packages baseline and submits benchmarking job to server
├── debug_local.sh            # validation pipeline locally without server submission
└── plot.sh                   # update plots
```

I feel like the repo is a bit of a bigger mess than I wanted it to be, but fuck it, we'll improve it along the way.
