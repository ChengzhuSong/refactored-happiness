#!/usr/bin/env python3
"""Re-encode per-element images with aspect-ratio-preserving CLIP letterboxing.

The source per-image parquet remains unchanged. For each split this script writes:

* ``crello_<split>_image_embeddings_letterbox.npy`` for resumable encoding.
* ``crello_<split>_elements_per_image_letterbox.parquet`` with only the
  ``image_embedding`` column replaced.
* ``crello_<split>_image_embeddings_letterbox.json`` with provenance and status.

The output parquet retains row order and all non-image-embedding columns from the
source. Run ``prepare_poster_inputs.py`` on it to create model-facing arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


CLIP_RGB_MEAN = (0.48145466, 0.4578275, 0.40821073)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode Crello element images with CLIP after square letterboxing."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--base-dir", type=Path, default=Path("data/crello"))
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4, help="Parallel image-loading workers")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--padding",
        choices=["clip-mean", "white", "black"],
        default="clip-mean",
        help="Letterbox canvas color; clip-mean becomes zero after CLIP normalization",
    )
    parser.add_argument("--suffix", default="letterbox")
    parser.add_argument("--parquet-batch-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Benchmark only the first N rows; does not write dataset artifacts",
    )
    return parser.parse_args()


def padding_rgb(name: str) -> tuple[int, int, int]:
    if name == "clip-mean":
        return tuple(round(value * 255) for value in CLIP_RGB_MEAN)
    if name == "white":
        return (255, 255, 255)
    return (0, 0, 0)


def resolve_image_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def load_letterboxed_image(path: Path, size: int, fill: tuple[int, int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, fill + (255,))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")

        image.thumbnail((size, size), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (size, size), fill)
        left = (size - image.width) // 2
        top = (size - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas


def load_batch(
    paths: Iterable[Path], size: int, fill: tuple[int, int, int], workers: int
) -> list[Image.Image]:
    paths = list(paths)
    if workers <= 1:
        return [load_letterboxed_image(path, size, fill) for path in paths]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda path: load_letterboxed_image(path, size, fill), paths))


def read_image_paths(source_path: Path, base_dir: Path) -> list[Path]:
    table = pq.read_table(source_path, columns=["image"])
    values = table.column("image").to_pylist()
    paths = [resolve_image_path(base_dir, value) for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        sample = "\n".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} element images are missing. First paths:\n{sample}")
    return paths


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_or_create_embeddings(
    path: Path, rows: int, dimension: int, state: dict, overwrite: bool
) -> tuple[np.memmap, int]:
    if overwrite:
        path.unlink(missing_ok=True)

    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != (rows, dimension) or array.dtype != np.float32:
            raise ValueError(
                f"Existing {path} has shape/dtype {array.shape}/{array.dtype}; "
                f"expected {(rows, dimension)}/float32. Use --overwrite to replace it."
            )
        next_index = int(state.get("next_index", 0))
        if not 0 <= next_index <= rows:
            raise ValueError(f"Invalid next_index={next_index} in state for {path}")
        return array, next_index

    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, dimension))
    return array, 0


def encode_paths(
    paths: list[Path],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
    workers: int,
    image_size: int,
    fill: tuple[int, int, int],
    output: np.ndarray | None = None,
    start_index: int = 0,
    progress_callback=None,
) -> np.ndarray | None:
    model.eval()
    started = time.perf_counter()
    processed = 0

    for start in range(start_index, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        images = load_batch(paths[start:end], image_size, fill, workers)
        inputs = processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixel_values)
            features = torch.nn.functional.normalize(features, dim=-1)
        values = features.cpu().numpy().astype(np.float32, copy=False)

        if output is not None:
            output[start:end] = values
        processed += end - start
        if progress_callback is not None:
            progress_callback(end)

        elapsed = time.perf_counter() - started
        rate = processed / max(elapsed, 1e-9)
        remaining = (len(paths) - end) / max(rate, 1e-9)
        print(
            f"  {end:>7}/{len(paths)} images | {rate:5.2f} images/s | "
            f"ETA {remaining / 60:6.1f} min",
            flush=True,
        )

    if output is None:
        return values
    return None


def embedding_list_array(values: np.ndarray, value_type: pa.DataType) -> pa.ListArray:
    rows, dimension = values.shape
    offsets = pa.array(np.arange(rows + 1, dtype=np.int32) * dimension)
    flat = pa.array(values.reshape(-1), type=value_type)
    return pa.ListArray.from_arrays(offsets, flat)


def rewrite_parquet_embeddings(
    source_path: Path,
    output_path: Path,
    embeddings: np.ndarray,
    batch_size: int,
    metadata: dict,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"Output parquet already exists, validating it: {output_path}")
        validate_output_parquet(source_path, output_path, embeddings)
        return

    source = pq.ParquetFile(source_path)
    schema = source.schema_arrow
    column_index = schema.get_field_index("image_embedding")
    if column_index < 0:
        raise KeyError(f"image_embedding column not found in {source_path}")
    embedding_type = schema.field(column_index).type
    if not pa.types.is_list(embedding_type):
        raise TypeError(f"Expected list image_embedding, found {embedding_type}")

    schema_metadata = dict(schema.metadata or {})
    schema_metadata[b"image_embedding_preprocessing"] = json.dumps(metadata).encode("utf-8")
    output_schema = schema.with_metadata(schema_metadata)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    offset = 0
    with pq.ParquetWriter(temporary, output_schema, compression="snappy") as writer:
        for batch in source.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            end = offset + table.num_rows
            replacement = embedding_list_array(
                np.asarray(embeddings[offset:end]), embedding_type.value_type
            )
            table = table.set_column(column_index, schema.field(column_index), replacement)
            writer.write_table(table)
            offset = end
            print(f"  wrote {offset:>7}/{source.metadata.num_rows} parquet rows", flush=True)

    if offset != source.metadata.num_rows or offset != len(embeddings):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Parquet rewrite row mismatch: wrote={offset}, source={source.metadata.num_rows}, "
            f"embeddings={len(embeddings)}"
        )
    os.replace(temporary, output_path)
    validate_output_parquet(source_path, output_path, embeddings)


def validate_output_parquet(
    source_path: Path, output_path: Path, embeddings: np.ndarray
) -> None:
    source = pq.ParquetFile(source_path)
    output = pq.ParquetFile(output_path)
    if output.metadata.num_rows != source.metadata.num_rows:
        raise ValueError("Output parquet row count differs from source")
    if output.schema_arrow.names != source.schema_arrow.names:
        raise ValueError("Output parquet columns differ from source")

    indices = sorted(set([0, len(embeddings) // 2, len(embeddings) - 1]))
    row_group_starts = np.cumsum(
        [0] + [output.metadata.row_group(i).num_rows for i in range(output.metadata.num_row_groups)]
    )
    for index in indices:
        row_group = int(np.searchsorted(row_group_starts, index, side="right") - 1)
        local_index = int(index - row_group_starts[row_group])
        selected = output.read_row_group(row_group, columns=["image_embedding"]).column(0)
        parquet_value = np.asarray(selected[local_index].as_py(), dtype=np.float32)
        if not np.array_equal(parquet_value, np.asarray(embeddings[index], dtype=np.float32)):
            difference = float(np.max(np.abs(parquet_value - embeddings[index])))
            raise ValueError(f"Embedding mismatch at row {index}; max abs diff={difference}")
    print(f"Validated {output_path}: rows, columns, and sampled embeddings match")


def process_split(
    split: str,
    args: argparse.Namespace,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> None:
    base_dir = args.base_dir.resolve()
    source_path = base_dir / f"crello_{split}_elements_per_image.parquet"
    embedding_path = base_dir / f"crello_{split}_image_embeddings_{args.suffix}.npy"
    output_path = base_dir / f"crello_{split}_elements_per_image_{args.suffix}.parquet"
    state_path = base_dir / f"crello_{split}_image_embeddings_{args.suffix}.json"

    if not source_path.exists():
        raise FileNotFoundError(source_path)
    print(f"Reading image paths for {split} from {source_path}")
    paths = read_image_paths(source_path, base_dir)

    if args.limit is not None:
        paths = paths[: args.limit]
        started = time.perf_counter()
        encode_paths(
            paths,
            model,
            processor,
            device,
            args.batch_size,
            args.workers,
            args.image_size,
            padding_rgb(args.padding),
        )
        elapsed = time.perf_counter() - started
        print(f"Benchmark: {len(paths)} images in {elapsed:.2f}s ({len(paths) / elapsed:.2f}/s)")
        return

    provenance = {
        "split": split,
        "source_parquet": str(source_path),
        "output_parquet": str(output_path),
        "embedding_file": str(embedding_path),
        "model_name": args.model_name,
        "method": "aspect-ratio-preserving square letterbox",
        "image_size": args.image_size,
        "padding": args.padding,
        "rows": len(paths),
        "embedding_dim": int(model.config.projection_dim),
    }
    state = {}
    if state_path.exists() and not args.overwrite:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = {key: provenance[key] for key in provenance if key not in {"output_parquet"}}
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"Existing state {state_path} does not match this run. Use --overwrite."
            )

    embeddings, next_index = load_or_create_embeddings(
        embedding_path,
        len(paths),
        int(model.config.projection_dim),
        state,
        args.overwrite,
    )
    if args.overwrite:
        state_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        next_index = 0

    def record_progress(index: int) -> None:
        embeddings.flush()
        save_json(state_path, {**provenance, "next_index": index, "complete": False})

    if next_index < len(paths):
        print(f"Encoding {split} from row {next_index} on {device}")
        encode_paths(
            paths,
            model,
            processor,
            device,
            args.batch_size,
            args.workers,
            args.image_size,
            padding_rgb(args.padding),
            output=embeddings,
            start_index=next_index,
            progress_callback=record_progress,
        )
        embeddings.flush()
    else:
        print(f"Embeddings already complete for {split}")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Non-finite embeddings found for {split}")
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            f"Embeddings are not L2 normalized for {split}: "
            f"norm range [{norms.min()}, {norms.max()}]"
        )

    save_json(state_path, {**provenance, "next_index": len(paths), "complete": True})
    print(f"Rewriting image_embedding into {output_path}")
    rewrite_parquet_embeddings(
        source_path,
        output_path,
        embeddings,
        args.parquet_batch_size,
        provenance,
        args.overwrite,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.image_size <= 0:
        raise ValueError("batch-size and image-size must be positive; workers cannot be negative")

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading {args.model_name} on {device}")
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device).eval()

    for split in args.splits:
        process_split(split, args, model, processor, device)


if __name__ == "__main__":
    main()
