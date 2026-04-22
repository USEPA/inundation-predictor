#!/usr/bin/env python3
"""
Recursively find GeoTIFFs by exact filename (with .tiff extension) and apply
internal ZSTD compression at level 9, keeping the .tiff extension. Runs in parallel.

Requires:
    - GDAL with ZSTD support (preferred), OR
    - rasterio (also requires GDAL/libtiff with ZSTD support)

Usage:
    python compress_geotiff_parallel.py ROOT_DIR TARGET_FILENAME.tiff
    # Options:
    #   --level 9              : ZSTD compression level (default: 9)
    #   --jobs 4               : Number of parallel workers (default: min(4, CPU))
    #   --no-inplace           : Write to a new file instead of replacing original
    #   --suffix _zstd         : Suffix for new files when not using in-place
    #   --case-insensitive     : Case-insensitive filename match
    #   --dry-run              : Only list matches
    #   --follow-symlinks      : Follow symlinked directories during search

Notes:
    - In-place mode is atomic: we write to a temp file next to the source, then replace.
    - For slow disks, using a small --jobs (e.g., 2–4) often performs best.
"""

import argparse
import os
import sys
import tempfile
from typing import Iterator, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

def find_matching_tiffs(root: str, target_name: str, case_insensitive: bool=False, follow_symlinks: bool=False) -> Iterator[str]:
    """
    Yield full paths to files whose basename exactly matches target_name
    and ends with .tiff (case-sensitive by default).
    """
    if not target_name.lower().endswith(".tiff"):
        target_name = target_name + ".tiff"

    if case_insensitive:
        target_name_cmp = target_name.lower()
        for dirpath, _dirs, files in os.walk(root, followlinks=follow_symlinks):
            for fn in files:
                if fn.lower() == target_name_cmp and fn.lower().endswith(".tiff"):
                    yield os.path.join(dirpath, fn)
    else:
        for dirpath, _dirs, files in os.walk(root, followlinks=follow_symlinks):
            for fn in files:
                if fn == target_name and fn.endswith(".tiff"):
                    yield os.path.join(dirpath, fn)

def compress_with_gdal(src: str, dst: str, level: int) -> None:
    from osgeo import gdal
    gdal.UseExceptions()
    opts = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=[
            "COMPRESS=ZSTD",
            f"ZSTD_LEVEL={level}",
            "TILED=YES",
            "BIGTIFF=IF_SAFER",
            # Optionally you can add NUM_THREADS=1 to avoid internal multi-threading
            # which may not help on a slow disk:
            # "NUM_THREADS=1"
        ]
    )
    ds = gdal.Translate(dst, src, options=opts)
    if ds is None:
        raise RuntimeError("gdal.Translate returned None")
    ds = None  # close/destroy

def compress_with_rasterio(src: str, dst: str, level: int) -> None:
    import rasterio
    with rasterio.Env():
        with rasterio.open(src) as src_ds:
            profile = src_ds.profile.copy()
            # Ensure we stay TIFF and apply ZSTD compression
            profile.update(
                driver="GTiff",
                compress="zstd",
                zstd_level=level,
                tiled=True,
                # You can add predictor=2 or 3 depending on data type to improve ratios
                # predictor=3 if floating-point, predictor=2 if integer
            )
            # Preserve block sizes if present
            if "blockxsize" in src_ds.profile and "blockysize" in src_ds.profile:
                profile.update(
                    blockxsize=src_ds.profile["blockxsize"],
                    blockysize=src_ds.profile["blockysize"],
                )
            with rasterio.open(dst, "w", **profile) as dst_ds:
                # Copy color table if present
                try:
                    cmap = src_ds.colormap(1)
                    if cmap:
                        dst_ds.write_colormap(1, cmap)
                except Exception:
                    pass
                # Copy data block-wise
                for _, window in dst_ds.block_windows():
                    data = src_ds.read(window=window)
                    dst_ds.write(data, window=window)
                # Copy tags
                dst_ds.update_tags(**src_ds.tags())
                for b in range(1, src_ds.count + 1):
                    dst_ds.update_tags(b, **src_ds.tags(b))

def compress_geotiff_zstd(src: str, level: int = 9, inplace: bool = True, suffix: str = "_zstd") -> str:
    """
    Rewrites a GeoTIFF with internal ZSTD compression. Returns the destination path.
    If inplace=True, uses a temporary file and atomically replaces the original.
    """
    # Destination path
    if inplace:
        dirn, base = os.path.split(src)
        fd, tmp_path = tempfile.mkstemp(prefix=base + ".", suffix=".tiff", dir=dirn)
        os.close(fd)  # GDAL/rasterio will reopen
        dst = tmp_path
    else:
        base, _ext = os.path.splitext(src)
        dst = f"{base}{suffix}.tiff"

    # Try GDAL first, then rasterio
    tried = []
    try:
        from osgeo import gdal  # noqa: F401
        tried.append("GDAL")
        compress_with_gdal(src, dst, level)
    except Exception as e_gdal:
        try:
            tried.append("rasterio")
            compress_with_rasterio(src, dst, level)
        except Exception as e_rio:
            # Clean temporary dst if created
            if os.path.exists(dst):
                try:
                    os.remove(dst)
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to compress with {', '.join(tried)}.\n"
                f"GDAL error: {e_gdal}\n"
                f"rasterio error: {e_rio}\n"
                "Please install GDAL with ZSTD support or rasterio."
            )

    # Replace original if requested
    if inplace:
        os.replace(dst, src)
        return src
    else:
        return dst

def _worker(task: Tuple[str, int, bool, str]) -> Tuple[str, Optional[str], bool, str]:
    """
    Worker to run in a separate process.

    Args:
        task: (src, level, inplace, suffix)

    Returns:
        (src, dst_or_none, success, message)
    """
    src, level, inplace, suffix = task
    try:
        dst = compress_geotiff_zstd(src, level=level, inplace=inplace, suffix=suffix)
        msg = "in-place" if inplace else dst
        return (src, None if inplace else dst, True, f"Compressed: {src} -> {msg}")
    except Exception as e:
        return (src, None, False, f"{src}: {e}")

def main():
    ap = argparse.ArgumentParser(description="Compress GeoTIFFs (.tiff) with internal ZSTD (level 9) in parallel.")
    ap.add_argument("root", help="Root directory to search")
    ap.add_argument("name", help="Exact filename to match (must end with .tiff, or it will be appended)")
    ap.add_argument("--level", type=int, default=9, help="ZSTD compression level (default: 9)")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                    help="Number of parallel workers (default: min(4, CPU))")
    ap.add_argument("--no-inplace", action="store_true", help="Write to a new .tiff instead of replacing original")
    ap.add_argument("--suffix", default="_zstd", help="Suffix for output name when not using in-place (default: _zstd)")
    ap.add_argument("--case-insensitive", action="store_true", help="Case-insensitive filename match")
    ap.add_argument("--follow-symlinks", action="store_true", help="Follow symlinked directories during search")
    ap.add_argument("--dry-run", action="store_true", help="Only list matches without compressing")
    args = ap.parse_args()

    inplace = not args.no_inplace
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Error: Root is not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    matches = list(find_matching_tiffs(root, args.name, args.case_insensitive, args.follow_symlinks))
    if not matches:
        print("No matching .tiff files found.")
        return

    print(f"Found {len(matches)} matching .tiff file(s).")
    if args.dry_run:
        for p in matches:
            print(p)
        return

    # Prepare tasks
    tasks = [(src, args.level, inplace, args.suffix) for src in matches]

    # Run in parallel
    jobs = max(1, args.jobs)
    print(f"Starting compression with {jobs} parallel worker(s)...")
    ok = 0
    err = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        future_to_src = {ex.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(future_to_src):
            src = future_to_src[fut]
            try:
                _src, _dst, success, message = fut.result()
                if success:
                    ok += 1
                    print(f"[OK] {message}")
                else:
                    err += 1
                    print(f"[ERR] {message}", file=sys.stderr)
            except Exception as e:
                err += 1
                print(f"[ERR] {src}: {e}", file=sys.stderr)

    print(f"Done. Compressed {ok} of {len(matches)} file(s). Errors: {err}")

if __name__ == "__main__":
    main()