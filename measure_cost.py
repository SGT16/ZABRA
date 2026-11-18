"""Checkpoint-based CUDA cost measurements for the CIFAR-100/ResNet18 ZABRA study."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile


ARMS = ("ZABRA", "ReLU")
BACKENDS = ("eager", "cudagraphs")
WORKLOADS = ("model_b1", "model_b128", "stem_forward", "stem_backward")
CELLS = {f"{arm}_{backend}_{work}": (arm, backend, work)
         for arm in ARMS for backend in BACKENDS for work in WORKLOADS}
WARMUP, BLOCKS, REPEATS = 20, 10, 20
INPUT_SEED = 20261114
DATASET_HASH = "eb587b75b3e0dca80d94a890ddf593166bedb43fbb6f20f28525b2819f0d7abb"


def check_pause(path):
    if path and path.is_file():
        raise SystemExit(75)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                     encoding="utf-8", suffix=".tmp") as stream:
        temp = Path(stream.name)
        json.dump(data, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


@contextmanager
def measurement_lock(folder):
    import msvcrt
    with (folder / "measure.lock").open("a+b") as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def load_model(torch, weights, arm):
    from activation_benchmark.models import build_model
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        ckpt = torch.load(weights, weights_only=True, map_location="cpu")
    meta = ckpt["metadata"]
    expected = dict(dataset="cifar100", dataset_fingerprint_sha256=DATASET_HASH, model="ResNet18",
                    activation=arm, activation_scope="full_backbone", protocol="full_train", epochs_run=100)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{weights}: expected {key}={value}, received {meta.get(key)}")
    model = build_model("ResNet18", 3, 100, arm, pretrained=False, small_input_stem=True)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.eval(), {**expected, "experiment_id": meta["experiment_id"], "seed": meta["seed"]}


def timed(torch, operation, *, prepare=None, pause_file=None):
    def run():
        if prepare is not None:
            prepare()
        operation()

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    blocks = []
    for _ in range(BLOCKS):
        check_pause(pause_file)
        n = REPEATS if prepare is not None else 1
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(n)]
        for first, last in events:
            first.record()
            last.record()
        torch.cuda.synchronize()
        if prepare is None:
            first, last = events[0]
            first.record()
            for _ in range(REPEATS):
                operation()
            last.record()
        else:
            for first, last in events:
                prepare()
                first.record()
                operation()
                last.record()
        torch.cuda.synchronize()
        value = sum(first.elapsed_time(last) for first, last in events) / REPEATS
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid CUDA event time {value}")
        blocks.append(value)
    q1, _, q3 = statistics.quantiles(blocks, n=4, method="inclusive")
    return dict(block_ms_per_call=blocks, median_ms=statistics.median(blocks),
                q1_ms=q1, q3_ms=q3, sample_sd_ms=statistics.stdev(blocks),
                allocated_after_warmup_bytes=baseline,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_increment_over_warmup_bytes=torch.cuda.max_memory_allocated() - baseline)


def capture_evidence(torch, operation):
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA profiling is unavailable")
    probe = torch.ones(1, device="cuda")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        operation()
        probe.neg_()
        torch.cuda.synchronize()
    events = list(prof.events())
    names = [e.name.casefold() for e in events]
    launch_api = any(s.startswith(("cudalaunchkernel", "culaunchkernel")) for s in names)
    if ("aten::neg_" not in names or not launch_api
            or not any(e.device_type == torch.autograd.DeviceType.CUDA for e in events)):
        raise RuntimeError("The profiler did not record CUDA kernel and launch activity for the eager probe")
    launches = sum(name.startswith(("cudagraphlaunch", "cugraphlaunch")) for name in names)
    regions = sorted({s for s in names if "torch-compiled region" in s or "compiledfunction" in s})
    return dict(launches=launches, compiled_regions=regions, cuda_probe_observed=True,
                backend_skip_count=int(torch._dynamo.utils.counters["inductor"]["cudagraph_skips"]))


def graph_exclusion(evidence):
    if evidence["backend_skip_count"] > 0:
        return "cudagraph_backend_reported_skip"
    if evidence["launches"] == 0 or not evidence["compiled_regions"]:
        return "cuda_graph_replay_not_verified"
    return None


def measure_cell(args, torch, arm, backend, workload):
    path = args.zabra_weights if arm == "ZABRA" else args.relu_weights
    model, meta = load_model(torch, path, arm)
    is_model = workload.startswith("model_")
    is_backward = workload == "stem_backward"
    batch = 1 if workload == "model_b1" else 128
    shape = (batch, 3, 32, 32) if is_model else (batch, 64, 32, 32)
    dtype = torch.float32 if is_model else torch.float16
    module = (model if is_model else model.relu).cuda()
    del model
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = False
    candidate = (module if backend == "eager" else
                 torch.compile(module, backend=backend, fullgraph=True, dynamic=False))
    generator = torch.Generator(device="cuda").manual_seed(INPUT_SEED)
    x = torch.randn(shape, device="cuda", dtype=dtype, generator=generator).requires_grad_(is_backward)
    tensors = (x, *module.parameters()) if is_backward else ()
    upstream = (torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * 0.01
                if is_backward else None)
    def forward(callable_module):
        with torch.set_grad_enabled(is_backward), torch.autocast("cuda", dtype=torch.float16):
            return callable_module(x)

    def check(expected, value):
        if not torch.isfinite(expected).all() or not torch.isfinite(value).all():
            raise ValueError("Nonfinite outputs or gradients in numerical equivalence check")
        torch.testing.assert_close(value, expected, rtol=0.05, atol=0.005)

    def check_numerics():
        reference = forward(module)
        if backend != "eager":
            torch.compiler.cudagraph_mark_step_begin()
        observed = forward(candidate)
        check(reference, observed)
        if is_backward:
            ref_grads = torch.autograd.grad(reference, tensors, upstream)
            obs_grads = torch.autograd.grad(observed, tensors, upstream)
            for a, b in zip(ref_grads, obs_grads, strict=True):
                check(a, b)

    check_numerics()
    prepared = None
    def start_forward():
        if backend != "eager":
            torch.compiler.cudagraph_mark_step_begin()
        return forward(candidate)

    def prepare_backward():
        nonlocal prepared
        prepared = start_forward()

    def backward():
        nonlocal prepared
        torch.autograd.grad(prepared, tensors, upstream)
        prepared = None

    op = backward if is_backward else start_forward
    prepare = prepare_backward if is_backward else None
    def proof():
        for _ in range(6):
            if prepare is not None:
                prepare()
            op()
        torch.cuda.synchronize()
        if prepare is not None:
            prepare()
        return capture_evidence(torch, op)

    device = torch.cuda.get_device_properties(0)
    evidence = {}
    row = dict(arm=arm, workload=workload, backend=backend, shape=shape, input_dtype=str(dtype),
               input_seed=INPUT_SEED, precision="cuda_amp_float16", checkpoint=meta, numerical_check="passed",
               warmup_calls=WARMUP, blocks=BLOCKS, calls_per_block=REPEATS, graph_evidence=evidence,
               gpu=device.name, gpu_uuid=str(device.uuid), torch=str(torch.__version__),
               cuda=torch.version.cuda, host=platform.node())
    if backend != "eager":
        evidence["before_timing"] = proof()
        reason = graph_exclusion(evidence["before_timing"])
        if reason:
            check_numerics()
            return dict(row, status="excluded", benchmark_accepted=False, reason=reason)
    result = timed(torch, op, prepare=prepare, pause_file=args.pause_file)
    if backend != "eager":
        evidence["after_timing"] = proof()
    check_numerics()
    reason = graph_exclusion(evidence["after_timing"]) if evidence else None
    if reason:
        return dict(row, status="excluded", benchmark_accepted=False, reason=reason)
    return dict(row, status="accepted", benchmark_accepted=True,
                images_per_second=batch * 1000 / result["median_ms"] if is_model else None, **result)


def summarize(cell_file, identity):
    cell = cell_file.stem
    arm, backend, work = CELLS[cell]
    row = json.loads(cell_file.read_text("utf-8"))
    if (row["identity"], row["cell"], row["arm"], row["backend"], row["workload"], row["numerical_check"]) != (
            identity, cell, arm, backend, work, "passed"):
        raise ValueError(f"Invalid measurement identity or numerics: {cell}")
    evidence = row["graph_evidence"]
    for phase in evidence.values():
        if (phase.get("cuda_probe_observed") is not True or phase["launches"] < 0
                or phase["backend_skip_count"] < 0 or not isinstance(phase["compiled_regions"], list)):
            raise ValueError(f"Invalid profiler evidence: {cell}")
    brief = dict(cell=cell, sha256=sha(cell_file), status=row["status"], benchmark_accepted=row["benchmark_accepted"])
    if row["status"] == "excluded" and row["benchmark_accepted"] is False:
        if backend == "cudagraphs" and set(evidence) in ({"before_timing"}, {"before_timing", "after_timing"}):
            final = evidence.get("after_timing", evidence["before_timing"])
            reason = graph_exclusion(final)
            if reason and row.get("reason") == reason and "median_ms" not in row:
                return dict(brief, reason=reason)
        raise ValueError(f"Exclusion lacks eligible proof: {cell}")
    expected_phases = set() if backend == "eager" else {"before_timing", "after_timing"}
    blocks = row.get("block_ms_per_call", [])
    if (row["status"] != "accepted" or row["benchmark_accepted"] is not True
            or set(evidence) != expected_phases or any(graph_exclusion(p) for p in evidence.values())
            or len(blocks) != BLOCKS or not all(0 < b < math.inf for b in blocks)
            or statistics.median(blocks) != row["median_ms"]):
        raise ValueError(f"Incomplete or invalid measurement: {cell}")
    return dict(brief, median_ms=row["median_ms"], peak_allocated_bytes=row["peak_allocated_bytes"],
                images_per_second=row["images_per_second"])


def run(args):
    check_pause(args.pause_file)
    import torch
    sys.path.insert(0, str(args.engine_dir.resolve()))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.cell:
        row = measure_cell(args, torch, *CELLS[args.cell])
        save(out / f"{args.cell}.json", {"identity": args.identity, "cell": args.cell, **row})
        return
    with measurement_lock(out):
        checkpoints = {}
        for arm, path in zip(ARMS, (args.zabra_weights, args.relu_weights), strict=True):
            model, metadata = load_model(torch, path, arm)
            checkpoints[arm] = dict(sha256=sha(path), **metadata)
            del model
        if checkpoints["ZABRA"]["seed"] != checkpoints["ReLU"]["seed"]:
            raise ValueError("ZABRA and ReLU checkpoint seeds differ")
        if args.validate_inputs:
            print(json.dumps(checkpoints, indent=2))
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.cuda.get_device_properties(0)
        recipe = dict(study="ZABRA CIFAR100 ResNet18 cost", checkpoints=checkpoints,
                      script_sha256=sha(Path(__file__)), engine_files={
                          name: sha(args.engine_dir / "activation_benchmark" / name)
                          for name in ("models.py", "activations.py")
                      }, host=platform.node(), gpu=device.name, gpu_uuid=str(device.uuid),
                      torch=str(torch.__version__), cudnn=torch.backends.cudnn.version(), cuda=torch.version.cuda,
                      checkpoint_site="relu (learned stem of small-input ResNet18)",
                      timing="warmed synthetic inputs on GPU, CUDA events, no data loader or transfer",
                      memory="fresh process per cell, peak allocated/reserved after warmup",
                      compile_backend="cudagraphs, fullgraph=True, dynamic=False",
                      warmup_calls=WARMUP, blocks=BLOCKS, calls_per_block=REPEATS)
        identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        protocol = out / "measurement_protocol.json"
        if protocol.is_file():
            if json.loads(protocol.read_text("utf-8")) != recipe:
                raise ValueError("Existing measurements use a different protocol, host or checkpoints")
        else:
            save(protocol, recipe)
        rows = []
        for cell in CELLS:
            check_pause(args.pause_file)
            cell_file = out / f"{cell}.json"
            if not cell_file.is_file():
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                        "--cell", cell, "--identity", identity])
                if child.returncode:
                    raise SystemExit(child.returncode)
            rows.append(summarize(cell_file, identity))
        accepted = [r["cell"] for r in rows if r["benchmark_accepted"]]
        excluded = [r["cell"] for r in rows if not r["benchmark_accepted"]]
        save(out / "measurement_complete.json", dict(status="complete", identity=identity,
                cells_completed=len(rows), cells_measured=len(accepted), cells_excluded=len(excluded),
                measured=accepted, excluded=excluded, results=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--zabra-weights", type=Path, required=True)
    parser.add_argument("--relu-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-inputs", action="store_true", help="Validate both checkpoints on CPU and exit")
    parser.add_argument("--pause-file", type=Path, help="Exit 75 on an operator pause; completed cells are retained")
    parser.add_argument("--cell", choices=CELLS, help=argparse.SUPPRESS)
    parser.add_argument("--identity", help=argparse.SUPPRESS)
    run(parser.parse_args())
