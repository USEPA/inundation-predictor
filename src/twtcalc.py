import os
import shutil
import datetime
import rioxarray
import xarray as xr
import zipfile
from rasterio.enums import Resampling
import numpy as np
import rasterio
from rasterio import warp
from osgeo import gdal
gdal.UseExceptions()

def calculate_strm_permanence(
    *,
    fname_perc_inundation=None,
    fname_strm_mask=None,
    verbose=False,
    overwrite=False,
    atol=1e-6 # tolerance for treating 100% as perennial
):
    """
    In-memory stream permanence computation

    Outputs (float32, NaN nodata embedded in pixel values):
      - perennial_strms_*.tiff: 1 where perennial (~100%), NaN elsewhere
      - nonperennial_strms_*.tiff: percent where 0 < perc < 100 - atol on streams, NaN elsewhere
    """
    if verbose:
        print("calling calculate_strm_permanence")

    if any(v is None for v in [fname_perc_inundation, fname_strm_mask]):
        raise ValueError("Required: fname_perc_inundation and fname_strm_mask")

    # Build output names, robust to .tif/.tiff
    base = os.path.basename(fname_perc_inundation)
    stem, _ = os.path.splitext(base)
    dstr = stem.replace("percent_inundated_grid_", "")
    out_dir = os.path.dirname(fname_perc_inundation)
    fname_p  = os.path.join(out_dir, f"perennial_strms_{dstr}.tiff")
    fname_np = os.path.join(out_dir, f"nonperennial_strms_{dstr}.tiff")

    if (os.path.isfile(fname_p) and os.path.isfile(fname_np)) and not overwrite:
        if verbose:
            print(f"found existing outputs:\n  {fname_p}\n  {fname_np}")
        return fname_p, fname_np

    if verbose:
        print(f"writing:\n  {fname_p}\n  {fname_np}")

    # Open rasters
    perc_da = rioxarray.open_rasterio(fname_perc_inundation, masked=True)
    mask_da = rioxarray.open_rasterio(fname_strm_mask, masked=True)

    # Ensure single band; drop band dimension if present
    if "band" in perc_da.dims:
        if perc_da.sizes["band"] != 1:
            raise ValueError("Percent-inundation raster must be single-band.")
        perc_da = perc_da.squeeze("band", drop=True)
    if "band" in mask_da.dims:
        if mask_da.sizes["band"] != 1:
            raise ValueError("Stream mask raster must be single-band.")
        mask_da = mask_da.squeeze("band", drop=True)

    # Align mask to percent grid using nearest-neighbor (categorical-safe)
    mask_da = mask_da.rio.reproject_match(perc_da, resampling="nearest")

    # Load fully into memory
    perc_da = perc_da.load()
    mask_da = mask_da.load()

    # Save georeferencing now (xarray ops can drop attrs)
    crs = perc_da.rio.crs
    try:
        transform = perc_da.rio.transform()
    except Exception:
        transform = None

    # Normalize nodata to NaN using nodata values from rioxarray
    nd_perc = perc_da.rio.nodata
    if nd_perc is not None and not (isinstance(nd_perc, float) and np.isnan(nd_perc)):
        perc_da = perc_da.where(perc_da != nd_perc, other=np.nan)

    nd_mask = mask_da.rio.nodata
    if nd_mask is not None and not (isinstance(nd_mask, float) and np.isnan(nd_mask)):
        mask_da = mask_da.where(mask_da != nd_mask, other=np.nan)

    # Cap any percent values > 100 to exactly 100 (preserve NaNs)
    perc_da = xr.where(perc_da > 100.0, 100.0, perc_da)

    # Build boolean stream mask: True where mask == 1 (NaNs -> False)
    stream_mask_bool = (mask_da == 1).fillna(False)

    # Restrict perc to streams; outside -> NaN
    perc_on_streams = perc_da.where(stream_mask_bool, other=np.nan)

    # Perennial where approx 100% within tolerance (NaNs evaluate to False)
    is_perennial = np.isfinite(perc_on_streams) & (np.abs(perc_on_streams - 100.0) <= atol)
    perennial_da = xr.where(is_perennial, 1.0, np.nan).astype("float32")

    # Non-perennial where on streams, finite, >0 and not perennial (use stream-masked perc)
    nonperennial_mask = (
        stream_mask_bool
        & np.isfinite(perc_on_streams)
        & (perc_on_streams > 0.0)
        & (~is_perennial)
    )
    nonperennial_da = xr.where(nonperennial_mask, perc_on_streams, np.nan).astype("float32")

    # Write georeferencing back only if we have it, and set nodata to NaN
    if crs is not None:
        perennial_da = perennial_da.rio.write_crs(crs, inplace=False)
        nonperennial_da = nonperennial_da.rio.write_crs(crs, inplace=False)
    else:
        if verbose:
            print("Warning: input percent grid has no CRS; writing outputs without CRS")

    if transform is not None:
        perennial_da = perennial_da.rio.write_transform(transform, inplace=False)
        nonperennial_da = nonperennial_da.rio.write_transform(transform, inplace=False)

    perennial_da = perennial_da.rio.write_nodata(np.nan, inplace=False)
    nonperennial_da = nonperennial_da.rio.write_nodata(np.nan, inplace=False)

    # Write outputs
    perennial_da.rio.to_raster(fname_p, compress="zstd", zlevel=9)
    nonperennial_da.rio.to_raster(fname_np, compress="zstd", zlevel=9)

    if verbose:
        print("done.")

    return fname_p, fname_np

def _read_base_grid_and_array(fname):
    """
    Read the base grid (TWI) as float32 and return:
    - arr: np.ndarray float32 (height, width), nodata -> np.nan
    - profile: rasterio profile with transform, crs, width, height
    """
    with rasterio.open(fname) as src:
        profile = src.profile.copy()
        arr = src.read(1, out_dtype="float32")
        nodata = src.nodata
        if nodata is not None and not np.isnan(nodata):
            arr[arr == nodata] = np.nan
        # Ensure single band float32; do not store NaN as nodata in metadata
        profile.update(
            count=1,
            dtype="float32",
        )
        # Remove nodata key if present; we will keep NaNs in data
        profile.pop("nodata", None)
    return arr, profile

def _reproject_to_target(src_path, dst_shape, dst_transform, dst_crs,
                         resampling=Resampling.bilinear, num_threads=None, dst_dtype="float32"):
    """
    Reproject a raster (first band) to a given target grid. Returns np.ndarray float32 with NaNs as nodata.
    """
    with rasterio.open(src_path) as src:
        dst = np.empty(dst_shape, dtype=dst_dtype)
        warp.reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
            num_threads=(num_threads or os.cpu_count() or 1),
        )
    return dst

def write_binary_inundation_tiff(
    fname,
    arr,
    profile,
    compress="zstd",
    bigtiff="IF_SAFER",
    blocksize=512,
    zlevel=9,          # higher level than 1: better compression, similar read speed
):
    """
    Write a binary inundation mask (1=water, 0=NoData) as a compact, fast GeoTIFF:
      - NBITS=1, tiled, ZSTD compression, sparse tiles
      - optional internal overviews with nearest resampling
    """
    # Ensure binary uint8 array with values {0, 1}
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8)
    unique_vals = np.unique(arr)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Array contains values other than 0/1: {unique_vals}")

    out_profile = profile.copy()
    # Force dtype and nodata for binary mask
    out_profile.update(
        driver="GTiff",
        dtype="uint8",
        nodata=0,
        tiled=True,
        compress=compress,
        BIGTIFF=bigtiff,
        blockxsize=blocksize,
        blockysize=blocksize,
        nbits=1,            # critical: 1-bit storage
        sparse_ok=True      # omit all-zero tiles
    )

    # Codec-specific options
    if compress in ("zstd", "deflate"):
        out_profile["zlevel"] = zlevel
        # predictor is not meaningful for 1-bit; leave unset

    # Clean None values
    out_profile = {k: v for k, v in out_profile.items() if v is not None}

    # Write dataset
    with rasterio.open(fname, "w", **out_profile) as dst:
        dst.write(arr, 1)

def calculate_inundation(*,
    dt_start: datetime.datetime,
    dt_end: datetime.datetime,
    wtd_raw_dir: str,
    inundation_out_dir: str,
    fname_twi: str,
    fname_twi_mean: str,
    fname_soil_trans: str,
    wtd_resampled_dir: str = None,
    verbose: bool = False,
    overwrite: bool = False,
    resampling=Resampling.bilinear,
    compress: str = "zstd",
    warp_threads: int = 4,
    blocksize: int = 512,
    zlevel: int = 9):

    if verbose:
        print('calling calculate_inundation')

    need = _check_exist(inundation_out_dir, dt_start, dt_end)
    if not need:
        if verbose:
            print(f' found existing inundation calculations in {inundation_out_dir}')
        return

    os.makedirs(inundation_out_dir, exist_ok=True)

    # Enable multi-threaded GDAL inside each reprojection
    gdal_env = rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", NUM_THREADS="ALL_CPUS")

    with gdal_env:
        # 1) Read/define target grid from TWI
        twi_arr, base_profile = _read_base_grid_and_array(fname_twi)
        height = base_profile['height']
        width  = base_profile['width']
        dst_transform = base_profile['transform']
        dst_crs = base_profile['crs']
        dst_shape = (height, width)

        # 2) Reproject twi_mean and soil_trans to target grid
        twi_mean_arr = _reproject_to_target(
            fname_twi_mean, dst_shape, dst_transform, dst_crs,
            resampling=resampling, num_threads=warp_threads, dst_dtype="float32"
        )
        soil_trans_arr = _reproject_to_target(
            fname_soil_trans, dst_shape, dst_transform, dst_crs,
            resampling=resampling, num_threads=warp_threads, dst_dtype="float32"
        )

        # 3) Compute threshold once: threshold = -(twi - twi_mean) / soil_trans, soil_trans != 0 else NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            threshold = np.where(
                soil_trans_arr != 0.0,
                -(twi_arr - twi_mean_arr) / soil_trans_arr,
                np.nan
            ).astype(np.float32)

        # 4) Sequentially process each day
        idt = dt_start
        while idt <= dt_end:
            dt_str = idt.strftime('%Y%m%d')
            fname_wtd_mean_raw = os.path.join(wtd_raw_dir, f'wtd_{dt_str}.tiff')
            fname_inund        = os.path.join(inundation_out_dir, f'inundation_{dt_str}.tiff')

            if not os.path.isfile(fname_wtd_mean_raw):
                raise FileNotFoundError(f'calculate_inundation could not find {fname_wtd_mean_raw}')

            if not os.path.isfile(fname_inund) or overwrite:
                if verbose:
                    print(f' processing {dt_str}')

                # Reproject WTD to target grid
                wtd_arr = _reproject_to_target(
                    fname_wtd_mean_raw, dst_shape, dst_transform, dst_crs,
                    resampling=resampling, num_threads=warp_threads, dst_dtype="float32"
                )

                # Apply logic: wtd_mean = -wtd; inundation = 1.0 where wtd_mean >= threshold, else NaN
                wtd_mean = -wtd_arr
                out = np.full(dst_shape, np.nan, dtype=np.float32)
                valid = (~np.isnan(wtd_mean)) & (~np.isnan(threshold))
                out[valid & (wtd_mean >= threshold)] = 1.0

                # Write result
                write_binary_inundation_tiff(
                    fname_inund, out, base_profile,
                    compress=compress, bigtiff="IF_SAFER",
                    blocksize=blocksize, zlevel=zlevel
                )

            idt += datetime.timedelta(days=1)

import os
import datetime
import numpy as np
import xarray as xr
import rioxarray

def calculate_summary_perc_inundated(
    *,
    dt_start,
    dt_end,
    inundation_raw_dir,
    inundation_summary_dir,
    fname_dem,
    verbose=False,
    overwrite=False,
    check_georef_once=True,
    compress="zstd",    # e.g., "LZW" for compression; None/"NONE" is fastest
    zlevel=9,          # higher level than 1: better compression, similar read speed
    tiled=True,
    blocksize=512,
):
    """
    Calculates percent of time inundated over the simulation period using a DEM-defined valid mask.

    Rules:
      - Valid mask: cells that are not masked (not nodata) in the DEM are valid for ALL days.
      - For each daily grid (uint8): convert all cells within the valid mask to 1 if they equal 1; else 0.
        (Cells outside the valid mask are ignored.)
      - Sum these 0/1 values into a uint32 accumulator.
      - percent = (inundated_days / total_days) * 100 for valid cells.
      - 0% is always written as NaN; cells outside the valid mask are NaN.

    Inputs:
      - dt_start (datetime): inclusive start date
      - dt_end   (datetime): exclusive end date
      - inundation_raw_dir (str): directory containing inundation_{YYYYMMDD}.tiff daily rasters (uint8)
      - inundation_summary_dir (str): output directory
      - fname_dem (str): DEM raster used to define the valid mask and provide shape/CRS/transform
    """
    # Validate
    if not (dt_start and dt_end and inundation_raw_dir and inundation_summary_dir and fname_dem):
        raise ValueError("Missing required kwarg(s): dt_start, dt_end, inundation_raw_dir, inundation_summary_dir, fname_dem.")
    if dt_end <= dt_start:
        raise ValueError("dt_end must be greater than dt_start (exclusive end).")

    os.makedirs(inundation_summary_dir, exist_ok=True)
    fname_output = os.path.join(
        inundation_summary_dir,
        f"percent_inundated_grid_{dt_start.strftime('%Y%m%d')}_to_{dt_end.strftime('%Y%m%d')}.tiff",
    )
    if os.path.isfile(fname_output) and not overwrite:
        if verbose:
            print(f"found existing summary percent inundation grid {fname_output}")
        return fname_output

    if verbose:
        print("calling calculate_summary_perc_inundated (DEM-valid mask; 1=inundated else 0 within mask)")
        print(f"writing summary percent inundation grid {fname_output}")

    # Load DEM to build valid mask and obtain georeferencing
    with rioxarray.open_rasterio(fname_dem, masked=True) as dem_da:
        dem = dem_da.sel(band=1).load()
    dem_mask = dem.isnull().values             # True where DEM is nodata -> invalid
    valid_mask = ~dem_mask                     # True where DEM is valid -> valid for all days
    height = dem.sizes["y"]
    width = dem.sizes["x"]
    dem_crs = dem.rio.crs
    dem_transform = dem.rio.transform()

    # Accumulator: uint32 count of inundated days
    inun_count = np.zeros((height, width), dtype=np.uint32)

    # Iterate dates [dt_start, dt_end)
    did_check_georef = False
    days_seen = 0
    idt = dt_start

    while idt < dt_end:
        dt_str = idt.strftime("%Y%m%d")
        f_in = os.path.join(inundation_raw_dir, f"inundation_{dt_str}.tiff")
        if not os.path.exists(f_in):
            raise FileNotFoundError(f"Missing daily inundation file: {f_in}")

        # Read daily raster as uint8 without masking
        with rioxarray.open_rasterio(f_in, masked=False) as da_in:
            in_da = da_in.sel(band=1)

            # One-time georeferencing check against DEM
            if check_georef_once and not did_check_georef:
                if in_da.rio.crs != dem_crs:
                    raise ValueError(f"CRS mismatch for {f_in}: expected {dem_crs}, got {in_da.rio.crs}")
                t_in = in_da.rio.transform()
                if any(abs(a - b) > 1e-9 for a, b in zip(t_in, dem_transform)):
                    raise ValueError(f"Transform mismatch for {f_in}: expected {dem_transform}, got {t_in}")
                did_check_georef = True

            arr = in_da.values  # numpy array, expected dtype uint8

        # Shape/dtype checks
        if arr.shape != (height, width):
            raise ValueError(f"Raster shape mismatch for {f_in}: expected {(height, width)}, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        # Build "inundated" mask: equal to 1, restricted to DEM valid mask.
        # We reuse the eq1 array to minimize temporaries.
        eq1 = (arr == 1)                 # bool
        np.logical_and(eq1, valid_mask, out=eq1)  # eq1 = eq1 & valid_mask

        # Increment inundation count by 1 where eq1 is True
        np.add(inun_count, 1, out=inun_count, where=eq1)

        # Cleanup per-iteration arrays
        del eq1, arr, in_da

        days_seen += 1
        if verbose and (days_seen % 25 == 0):
            print(f"  processed {days_seen} rasters (latest: {dt_str})")

        idt += datetime.timedelta(days=1)

    if days_seen == 0:
        raise ValueError("Empty date range (no days between dt_start and dt_end).")

    # Compute percentage over total days for valid cells; force 0% and invalid cells to NaN
    perc = np.empty((height, width), dtype=np.float32)
    # Start by setting all to NaN
    perc[:] = np.nan
    # For valid cells, compute (count / days_seen) * 100
    if np.any(valid_mask):
        perc[valid_mask] = (inun_count[valid_mask].astype(np.float32) / float(days_seen)) * 100.0
    # Force 0% to NaN as required
    perc[perc <= 0.0] = np.nan

    # Build output DataArray with georeferencing
    perc_da = xr.DataArray(perc, dims=("y", "x"), name="percent_inundated")
    perc_da = perc_da.rio.write_crs(dem_crs, inplace=False)
    perc_da = perc_da.rio.write_transform(dem_transform, inplace=False)
    perc_da.rio.write_nodata(np.nan, inplace=True)

    # Clean encoding to avoid serialization conflicts
    for key in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
        perc_da.attrs.pop(key, None)
        perc_da.encoding.pop(key, None)
    perc_da.encoding = {}

    # Write to disk
    write_kwargs = {}
    if compress is not None:
        write_kwargs["compress"] = compress
    if zlevel is not None:
        write_kwargs["zlevel"] = zlevel
    if tiled:
        write_kwargs.update(dict(tiled=True,
                                 blockxsize=blocksize, 
                                 blockysize=blocksize))
    perc_da.rio.to_raster(fname_output, **write_kwargs)

    if verbose:
        print(f"wrote {fname_output} (days={days_seen}, accumulator=uint32, 0%->NaN, DEM-valid mask)")

    return fname_output

def _zip_inundation(raw_dir, compression=zipfile.ZIP_LZMA, compresslevel=9):
    """
    Create a ZIP archive of 'raw_dir' including the directory itself as the top-level entry,
    then remove the original directory.

    Parameters:
        raw_dir       : path to the 'raw' directory
        compression   : zipfile compression (e.g., ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA)
        compresslevel : int or None; higher is smaller/slower (supported for DEFLATED/BZIP2/LZMA)

    Returns:
        - Path to the created ZIP file (string) if a ZIP is created.
        - None if the directory does not exist or is empty.

    Notes:
        - ZIP_DEFLATED is widely compatible; BZIP2/LZMA can compress smaller but may not open in all tools.
    """
    abs_raw = os.path.abspath(raw_dir)
    parent = os.path.dirname(abs_raw)
    zip_path = abs_raw + '.zip'
    kwargs = {}
    if compresslevel is not None:
        kwargs['compresslevel'] = compresslevel
    with zipfile.ZipFile(zip_path, mode='w', compression=compression, **kwargs) as zf:
        for dirpath, dirnames, filenames in os.walk(abs_raw):
            rel_dir = os.path.relpath(dirpath, parent)
            zf.write(dirpath, arcname=rel_dir)
            for name in filenames:
                full_path = os.path.join(dirpath, name)
                arcname = os.path.relpath(full_path, parent)
                zf.write(full_path, arcname=arcname)
    shutil.rmtree(abs_raw)
    return zip_path

def _check_exist(inundation_out_dir:str,dt_start:datetime.datetime,dt_end:datetime.datetime):
    idt = dt_start
    while idt <= dt_end:
        dt_str = idt.strftime('%Y%m%d')
        fname  = f'inundation_{dt_str}.tiff'
        if not os.path.isfile(os.path.join(inundation_out_dir,fname)):
            return True
        idt += datetime.timedelta(days=1)
    return False