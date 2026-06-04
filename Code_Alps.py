# Code_Alps.py
# Dr. T. Banerjee, banerjee@iau.uni-frankfurt.de
# Distributed Alps ENUFFT/CSA sweep driver.

import csv
import itertools
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from Module_Alps import (
    alps_metric_version,
    build_alps_case,
    build_alps_mesh,
    build_alps_modes_rows,
    build_alps_spectra_rows,
    deplane_dem_on_mesh,
    init_alps_worker,
    load_alps_dem,
    make_alps_config_tag,
    make_alps_sweep_tag,
    normalize_mesh_name,
    preprocessed_dem,
    process_alps_triangle,
    scalar_alps_result_row,
    summarise_alps_results,
    window_strategies,
)
from Module_Csv import write_dict_rows_csv


csv_dir = Path("./csv")
per_triangle_prefix = csv_dir / "Banerjee_2026_Enufft_Alps_PerTriangle"
summary_prefix = csv_dir / "Banerjee_2026_Enufft_Alps_Summary"
spectra_prefix = csv_dir / "Banerjee_2026_Enufft_Alps_Spectra"
modes_prefix = csv_dir / "Banerjee_2026_Enufft_Alps_Modes"
sweep_summary_prefix = csv_dir / "Banerjee_2026_Enufft_Alps_SweepSummary"
distributed_prefix = csv_dir / f"Banerjee_2026_Enufft_Alps_Distributed_{os.environ.get('SLURM_JOB_ID', 'local')}"
mesh_names = ("r2b4", "r2b5")
sweep_windows = tuple(window_strategies)
sweep_weights = ("uniform", "voronoi")
sweep_expansions = (1.0, 1.5, 2.0)
sweep_oversamples = (1.25, 1.5, 2.0)
spectra_rank_count = 40
max_triangles = None
overwrite_existing = False


# Print one progress line immediately.
def progress(message=""):
    print(message, flush=True)


# Print the aggregate numerical summary.
def print_summary(summary):
    progress("")
    progress("Summary")
    progress(f"  Triangles processed {summary['n_triangles']}")
    progress(f"  Median relative RMSE ENUFFT {summary['rel_rmse_en_median']:.3f}")
    progress(f"  Median relative RMSE CSA {summary['rel_rmse_csa_median']:.3f}")
    progress(f"  Median Parseval ratio ENUFFT {summary['var_ratio_en_median']:.3f}")
    progress(f"  Median Parseval ratio CSA {summary['var_ratio_csa_median']:.3f}")
    progress(f"  Median K* ENUFFT {summary['K_star_median']:.1f}")
    progress(f"  Median CSA unique pairs {summary['csa_pairs_median']:.1f}")
    progress(f"  Median window points outside target triangle {summary['n_encroached_points_median']:.0f}")


# Write per-triangle, summary, sorted-spectra, and sparse-mode CSV files.
def write_alps_outputs(good, summary, case, tag, spectra_rank_count):
    csv_dir.mkdir(parents=True, exist_ok=True)
    per_triangle_path = per_triangle_prefix.with_name(per_triangle_prefix.name + f"{tag}.csv")
    summary_path = summary_prefix.with_name(summary_prefix.name + f"{tag}.csv")
    spectra_path = spectra_prefix.with_name(spectra_prefix.name + f"{tag}.csv")
    modes_path = modes_prefix.with_name(modes_prefix.name + f"{tag}.csv")
    scalar_rows = [scalar_alps_result_row(row) for row in good]
    spectra_rows = []
    modes_rows = []
    for row in good:
        spectra_rows.extend(build_alps_spectra_rows(row, spectra_rank_count))
        modes_rows.extend(build_alps_modes_rows(case, row, "enufft", row["spectrum_enufft"]))
        if bool(row.get("csa_valid", False)):
            modes_rows.extend(build_alps_modes_rows(case, row, "csa", row["spectrum_csa"]))
    write_dict_rows_csv(scalar_rows, per_triangle_path)
    write_dict_rows_csv([summary], summary_path)
    write_dict_rows_csv(spectra_rows, spectra_path)
    write_dict_rows_csv(modes_rows, modes_path)
    progress(f"  Wrote {per_triangle_path}")
    progress(f"  Wrote {summary_path}")
    progress(f"  Wrote {spectra_path}")
    progress(f"  Wrote {modes_path}")


# Write the aggregate sweep summary CSV.
def write_sweep_summary(rows, tag):
    if not rows:
        return None
    csv_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = sweep_summary_prefix.with_name(sweep_summary_prefix.name + f"{tag}.csv")
    write_dict_rows_csv(rows, sweep_path)
    progress(f"  Wrote {sweep_path}")
    return sweep_path


# Write one pickle atomically so the reducer never reads a half-written chunk.
def write_pickle_atomic(value, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(output_path)


# Return the distributed coordination path for one suffix.
def distributed_path(args, suffix):
    return args.prefix.with_name(args.prefix.name + suffix)


# Read one existing summary CSV when a completed config is skipped.
def read_existing_summary(summary_path):
    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return dict(row)
    return None


# Parse the requested mesh presets.
def requested_meshes(args):
    names = [args.mesh] if args.mesh is not None else list(args.meshes)
    meshes = []
    for name in names:
        mesh_name = normalize_mesh_name(name)
        if mesh_name not in meshes:
            meshes.append(mesh_name)
    if not meshes:
        raise ValueError("At least one Alps mesh must be requested")
    return meshes


# Build the fixed distributed sweep settings.
def build_settings():
    return SimpleNamespace(
        prefix=distributed_prefix,
        mesh=None,
        meshes=mesh_names,
        cell_size=None,
        n_modes=None,
        window="square_centroid",
        expansion=2.0,
        oversample=1.5,
        weight="uniform",
        csa_sparse_modes=None,
        csa_chunk_size=512,
        max_triangles=max_triangles,
        spectra_ranks=spectra_rank_count,
        sweep=True,
        sweep_windows=sweep_windows,
        sweep_weights=sweep_weights,
        sweep_expansions=sweep_expansions,
        sweep_oversamples=sweep_oversamples,
        overwrite=overwrite_existing,
    )


# Build one base case for a mesh preset.
def build_base_case(args, mesh_name):
    cell_size = 80.0 if args.cell_size is None else float(args.cell_size)
    return build_alps_case({
        "mesh_name": mesh_name,
        "cell_size_km": cell_size,
        "n_modes": args.n_modes,
        "oversample": args.oversample,
        "window_expansion": args.expansion,
        "window_strategy": args.window,
        "weight_type": args.weight,
        "csa_sparse_modes": args.csa_sparse_modes,
        "csa_chunk_size": args.csa_chunk_size,
        "max_triangles": args.max_triangles,
    })


# Build all requested case dictionaries and stable output tags.
def build_case_records(args):
    records = []
    config_index = 1
    for mesh_name in requested_meshes(args):
        base_case = build_base_case(args, mesh_name)
        if args.sweep:
            combinations = itertools.product(
                args.sweep_windows,
                args.sweep_weights,
                args.sweep_expansions,
                args.sweep_oversamples,
            )
        else:
            combinations = [(base_case["window_strategy"], base_case["weight_type"], base_case["window_expansion"], base_case["oversample"])]
        for window, weight, expansion, oversample in combinations:
            case = build_alps_case({**base_case, "window_strategy": window, "weight_type": weight, "window_expansion": float(expansion), "oversample": float(oversample)})
            tag = make_alps_config_tag(case)
            if case["max_triangles"] is not None:
                tag += f"_first{case['max_triangles']}"
            records.append({"config_index": config_index, "mesh_name": case["mesh_name"], "tag": tag, "case": case})
            config_index += 1
    return records


# Return the final summary path for one case tag.
def summary_path_for_tag(tag):
    return summary_prefix.with_name(summary_prefix.name + f"{tag}.csv")


# Keep only the DEM arrays needed by Slurm workers.
def worker_dem_payload(dem_data):
    keys = [
        "dem_xy",
        "dem_h_deplaned",
        "mesh_triangle_id",
        "mesh_triangle_means",
        "mesh_triangle_counts",
        "Lx",
        "Ly",
        "lon_ref",
        "lat_ref",
        "smooth_km",
    ]
    return {key: dem_data[key] for key in keys if key in dem_data}


# Build and cache the deplaned DEM payload for one mesh.
def prepare_mesh_payload(args, dem_data, mesh_name):
    case = build_base_case(args, mesh_name)
    mesh = build_alps_mesh(dem_data, case["mesh_name"], case["cell_size_km"])
    mesh_dem = deplane_dem_on_mesh(dem_data, mesh)
    triangle_ids = list(range(len(mesh["triangles"])))
    if args.max_triangles is not None:
        triangle_ids = triangle_ids[: int(args.max_triangles)]
    cache_file = distributed_path(args, f"_Mesh_{mesh_name}.pkl")
    write_pickle_atomic({"dem_data": worker_dem_payload(mesh_dem), "mesh": mesh}, cache_file)
    nonempty = int((mesh_dem["mesh_triangle_counts"] > 0).sum())
    return {
        "mesh_name": mesh_name,
        "mesh_kind": mesh.get("mesh_kind", "unknown"),
        "mesh_version": mesh.get("mesh_version", ""),
        "mesh_signature": mesh.get("mesh_signature", ""),
        "nominal_dx_km": float(case["nominal_dx_km"]),
        "n_modes": int(case["n_modes"]),
        "n_mesh_vertices": int(len(mesh["vertices"])),
        "n_mesh_triangles": int(len(mesh["triangles"])),
        "n_nonempty_triangles": nonempty,
        "triangle_ids": triangle_ids,
        "cache_file": str(cache_file),
        "sweep_tag": make_alps_sweep_tag(case) + (f"_first{case['max_triangles']}" if case["max_triangles"] is not None else ""),
    }


# Generate the deterministic task table read by every Slurm rank.
def prepare_tasks(args):
    csv_dir.mkdir(parents=True, exist_ok=True)
    for path in csv_dir.glob(f"{args.prefix.name}_*"):
        if path.is_file():
            path.unlink()

    progress("Preparing distributed Alps task table")
    progress(f"  DEM {preprocessed_dem}")
    progress(f"  Meshes {', '.join(requested_meshes(args))}")
    dem_data = load_alps_dem()
    records = build_case_records(args)
    mesh_records = {}
    for mesh_name in requested_meshes(args):
        mesh_record = prepare_mesh_payload(args, dem_data, mesh_name)
        mesh_records[mesh_name] = mesh_record
        progress(f"  Mesh {mesh_name} ({mesh_record['mesh_kind']}, {mesh_record['mesh_version']}, {mesh_record['mesh_signature']})")
        progress(f"    Vertices {mesh_record['n_mesh_vertices']}, triangles {mesh_record['n_mesh_triangles']}, nonempty {mesh_record['n_nonempty_triangles']}")
        progress(f"    Triangles per config {len(mesh_record['triangle_ids'])}, N_modes {mesh_record['n_modes']}")

    active_configs = []
    skipped_configs = []
    tasks = []
    for record in records:
        summary_path = summary_path_for_tag(record["tag"])
        if summary_path.exists() and summary_path.stat().st_size > 0 and not args.overwrite:
            existing_summary = read_existing_summary(summary_path)
            expected_signature = mesh_records[record["mesh_name"]].get("mesh_signature", "")
            if existing_summary is not None and existing_summary.get("mesh_signature", "") == expected_signature and existing_summary.get("metric_version", "") == alps_metric_version:
                skipped_configs.append(record)
                continue
        active_configs.append(record)
        for triangle_id in mesh_records[record["mesh_name"]]["triangle_ids"]:
            tasks.append({
                "task_id": len(tasks),
                "config_index": int(record["config_index"]),
                "mesh_name": record["mesh_name"],
                "tag": record["tag"],
                "triangle_id": int(triangle_id),
            })

    task_table = distributed_path(args, "_Tasks.tsv")
    with task_table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "config_index", "mesh_name", "tag", "triangle_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(tasks)

    manifest = {
        "created_unix": time.time(),
        "dem": str(preprocessed_dem),
        "meshes": mesh_records,
        "spectra_ranks": int(args.spectra_ranks),
        "active_configs": active_configs,
        "skipped_configs": skipped_configs,
        "task_count": len(tasks),
    }
    with distributed_path(args, "_Manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    distributed_path(args, "_TaskCount.txt").write_text(f"{len(tasks)}\n")

    progress(f"  Domain {dem_data['Lx']:.1f} by {dem_data['Ly']:.1f} km")
    progress(f"  Configs total {len(active_configs) + len(skipped_configs)}")
    progress(f"  Configs skipped existing {len(skipped_configs)}")
    progress(f"  Configs active {len(active_configs)}")
    progress(f"  Triangle tasks active {len(tasks)}")
    progress(f"  Task table {task_table}")


# Load the manifest generated by prepare mode.
def load_manifest(args):
    with distributed_path(args, "_Manifest.json").open() as handle:
        return json.load(handle)


# Load the deterministic task table generated by prepare mode.
def load_tasks(args):
    with distributed_path(args, "_Tasks.tsv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            {
                "task_id": int(row["task_id"]),
                "config_index": int(row["config_index"]),
                "mesh_name": row["mesh_name"],
                "tag": row["tag"],
                "triangle_id": int(row["triangle_id"]),
            }
            for row in reader
        ]


# Load one prepared mesh payload for a worker rank.
def load_mesh_payload(mesh_record):
    with Path(mesh_record["cache_file"]).open("rb") as handle:
        return pickle.load(handle)


# Compute all tasks assigned to this Slurm rank by deterministic striding.
def run_worker(args):
    manifest = load_manifest(args)
    tasks = load_tasks(args)
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_tasks = [task for task in tasks if task["task_id"] % world_size == rank]
    progress(f"[rank {rank}] assigned {len(local_tasks)} of {len(tasks)} tasks")
    if not local_tasks:
        write_pickle_atomic([], distributed_path(args, f"_Rank_{rank:06d}.pkl"))
        return

    case_by_index = {int(record["config_index"]): record for record in manifest["active_configs"]}
    mesh_cache = {}
    current_context = None
    rows = []
    start_time = time.time()
    for local_index, task in enumerate(local_tasks, start=1):
        mesh_name = task["mesh_name"]
        if mesh_name not in mesh_cache:
            mesh_cache[mesh_name] = load_mesh_payload(manifest["meshes"][mesh_name])
            progress(f"[rank {rank}] loaded mesh {mesh_name} with {manifest['meshes'][mesh_name]['n_mesh_triangles']} triangles")
        record = case_by_index[int(task["config_index"])]
        context_key = (mesh_name, int(task["config_index"]))
        if current_context != context_key:
            payload = mesh_cache[mesh_name]
            init_alps_worker(payload["dem_data"], payload["mesh"], record["case"])
            current_context = context_key
        task_start = time.time()
        row = process_alps_triangle(int(task["triangle_id"]))
        rows.append({"config_index": int(task["config_index"]), "mesh_name": mesh_name, "tag": task["tag"], "triangle_id": int(task["triangle_id"]), "row": row})
        status = "SKIP" if row.get("skip", True) else "OK"
        progress(
            f"[rank {rank}] task {task['task_id'] + 1}/{len(tasks)} "
            f"local {local_index}/{len(local_tasks)} mesh {mesh_name} config {task['config_index']} "
            f"tri {task['triangle_id']} {status} {time.time() - task_start:.1f}s"
        )
    write_pickle_atomic(rows, distributed_path(args, f"_Rank_{rank:06d}.pkl"))
    progress(f"[rank {rank}] wrote {len(rows)} rows in {(time.time() - start_time) / 60.0:.1f} min")


# Reduce all rank chunks into the final CSV products.
def reduce_chunks(args):
    manifest = load_manifest(args)
    rows_by_config = {int(record["config_index"]): [] for record in manifest["active_configs"]}
    chunks = sorted(csv_dir.glob(f"{args.prefix.name}_Rank_*.pkl"))
    progress("Reducing distributed Alps chunks")
    progress(f"  Chunks found {len(chunks)}")
    for chunk_path in chunks:
        with chunk_path.open("rb") as handle:
            for item in pickle.load(handle):
                rows_by_config[int(item["config_index"])].append(item["row"])

    sweep_rows_by_mesh = {mesh_name: [] for mesh_name in manifest["meshes"]}
    for skipped in manifest["skipped_configs"]:
        summary = read_existing_summary(summary_path_for_tag(skipped["tag"]))
        if summary is not None:
            sweep_rows_by_mesh[skipped["mesh_name"]].append((int(skipped["config_index"]), summary))

    for record in manifest["active_configs"]:
        config_index = int(record["config_index"])
        mesh_name = record["mesh_name"]
        case = record["case"]
        tag = record["tag"]
        expected_triangles = [int(value) for value in manifest["meshes"][mesh_name]["triangle_ids"]]
        expected_count = len(expected_triangles)
        results = rows_by_config[config_index]
        present = {int(row.get("tri_num", -1)) for row in results}
        missing = [tri for tri in expected_triangles if tri not in present]
        if missing:
            raise RuntimeError(f"Config {config_index} {tag} is missing {len(missing)} triangle rows; first missing {missing[:10]}")
        if len(results) != expected_count:
            raise RuntimeError(f"Config {config_index} {tag} has {len(results)} rows, expected {expected_count}")
        expected_signature = manifest["meshes"][mesh_name].get("mesh_signature", "")
        bad_signature_count = sum(1 for row in results if row.get("mesh_signature", expected_signature) != expected_signature)
        if bad_signature_count:
            raise RuntimeError(f"Config {config_index} {tag} has {bad_signature_count} rows from a different mesh signature")
        results.sort(key=lambda row: int(row.get("tri_num", -1)))
        good = [row for row in results if not row.get("skip", True)]
        if not good:
            progress(f"  Config {config_index} {tag} has no good triangles; skipping final CSV")
            continue
        summary = summarise_alps_results(good, case)
        progress("")
        progress(f"[{config_index}] REDUCE {tag}")
        print_summary(summary)
        write_alps_outputs(good, summary, case, tag, int(manifest["spectra_ranks"]))
        sweep_rows_by_mesh[mesh_name].append((config_index, summary))

    for mesh_name, sweep_rows in sweep_rows_by_mesh.items():
        ordered_rows = [row for _, row in sorted(sweep_rows, key=lambda item: item[0])]
        if ordered_rows:
            write_sweep_summary(ordered_rows, manifest["meshes"][mesh_name]["sweep_tag"])
    progress("Distributed reduction complete")


# Select the distributed driver mode.
def selected_mode():
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "worker", "reduce"):
        raise SystemExit("Usage: python3 Code_Alps.py prepare|worker|reduce")
    return sys.argv[1]


# Dispatch one distributed driver mode.
def main():
    mode = selected_mode()
    args = build_settings()
    if mode == "prepare":
        prepare_tasks(args)
    elif mode == "worker":
        run_worker(args)
    elif mode == "reduce":
        reduce_chunks(args)
    else:
        raise RuntimeError("No mode selected")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
