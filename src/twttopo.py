import os
import rasterio
import py3dep
import geopandas
import rioxarray
import whitebox
import glob
import subprocess
from osgeo import gdal
from geocube.api.core import make_geocube
import math
import numpy as np
from affine import Affine
from rasterio.warp import transform_bounds

#from whitebox_workflows import WbEnvironment

async def download_dem(**kwargs):
    domain        = kwargs.get('domain',    None)
    verbose       = kwargs.get('verbose',   False)
    overwrite     = kwargs.get('overwrite', False)
    dem_rez       = kwargs.get('dem_rez',   None)
    fname_dem     = kwargs.get('fname_dem', None)
    if not os.path.isfile(fname_dem) or overwrite:
        avail = py3dep.check_3dep_availability(bbox=tuple(domain.total_bounds),crs=domain.crs)
        vals  = list()
        for k in avail.keys():
            try: vals.append(int(k.replace('m','')))
            except: pass
        if len(vals) == 0:
            raise ValueError(f'download_dem 3dep could not find dem resolution')
        rez = min(vals)
        if dem_rez in vals: rez = dem_rez # override with user input if available
        if verbose:
            print(f' downloading {str(rez)}m DEM to {fname_dem}')
        dem = py3dep.get_dem(geometry   = domain.geometry.union_all(),
                             resolution = rez,
                             crs        = domain.crs)
        dem.rio.to_raster(fname_dem)
    else:
        if verbose: print(f' using existing dem {fname_dem}')

def break_dem_old(**kwargs):
    fname_dem_parent = kwargs.get('fname_dem_parent', None)
    fname_dem_child  = kwargs.get('fname_dem_child',  None)
    fname_boundary   = kwargs.get('fname_boundary',       None)
    verbose          = kwargs.get('verbose',   False)
    overwrite        = kwargs.get('overwrite', False)
    if verbose: print(f'calling break_dem')
    if not os.path.isfile(fname_dem_child) or overwrite:
        if verbose: print(f' breaking {fname_dem_parent} using {fname_boundary} and writing to {fname_dem_child}')
        boundary = geopandas.read_file(fname_boundary)
        with rioxarray.open_rasterio(fname_dem_parent, 
                                    masked=True, 
                                    chunks="auto", 
                                    parse_coordinates=False) as parent_rds:
            boundary = boundary.to_crs(parent_rds.rio.crs)  
            minx, miny, maxx, maxy = boundary.total_bounds
            parent_rds2 = parent_rds.rio.clip_box(minx, miny, maxx, maxy)
            child_rds = parent_rds2.rio.clip(boundary.geometry.values, boundary.crs,from_disk=True)
            child_rds.rio.to_raster(fname_dem_child, driver="GTiff",compress="DEFLATE", tiled=True)
    else:
        if verbose: print(f' using existing dem {fname_dem_child}')

def break_dem(**kwargs):
    fname_dem_parent = kwargs.get('fname_dem_parent', None)
    fname_dem_child  = kwargs.get('fname_dem_child',  None)
    fname_boundary   = kwargs.get('fname_boundary',       None)
    verbose          = kwargs.get('verbose',   False)
    overwrite        = kwargs.get('overwrite', False)
    if verbose: print('calling break_dem')
    if not os.path.isfile(fname_dem_child) or overwrite:
        if verbose: print(f' creating {fname_dem_child} from {fname_dem_parent}')
        gdal.UseExceptions()
        src_ds = gdal.Open(fname_dem_parent, gdal.GA_ReadOnly)
        if src_ds is None: raise RuntimeError(f"Failed to open VRT: {fname_dem_parent}")
        warp_kwargs = {
            "format": "GTiff",
            "cutlineDSName": fname_boundary,      
            "cropToCutline": True,                
            "resampleAlg": "near",             
            "multithread": True,
            "warpOptions": ["NUM_THREADS=ALL_CPUS"],
            "warpMemoryLimit": 512 * 1024 * 1024,
            "creationOptions": [
                "TILED=YES",
                "COMPRESS=LZW",
                "BIGTIFF=IF_SAFER",
            ],
        }
        try:
            nodata = src_ds.GetRasterBand(1).GetNoDataValue()
            warp_kwargs["dstNodata"] = nodata
        except Exception:
            pass
        out_ds = gdal.Warp(fname_dem_child, src_ds, **warp_kwargs)
        if out_ds is None: raise RuntimeError("GDAL Warp failed")
        out_ds = None
        src_ds = None
    else:
        if verbose: print(f' found {fname_dem_child}')

def breach_dem(**kwargs):
    fname_dem_breached = kwargs.get('fname_dem_breached', None)
    fname_dem          = kwargs.get('fname_dem',          None)
    verbose            = kwargs.get('verbose',            False)
    overwrite          = kwargs.get('overwrite',          False)
    if verbose: print('calling breach_dem')
    if fname_dem is None:
        raise KeyError('breach_dem missing required argument fname_dem')
    if not os.path.isfile(fname_dem):
        raise ValueError(f'breach_dem could not find fname_dem {fname_dem}')
    if not os.path.isfile(fname_dem_breached) or overwrite:
        if verbose: print(f' using whitebox to breach dem and writing to {fname_dem_breached}')
        wbt = whitebox.WhiteboxTools()
        fname_filled = fname_dem.replace('.tif','_fill.tif')
        wbt.breach_single_cell_pits(
            dem=fname_dem, 
            output=fname_filled, 
        )
        try:
            cmd = [
                os.path.join(wbt.exe_path,wbt.exe_name),
                "-r=breach_depressions_least_cost",
                f"--dem={fname_filled}",
                f"--output={fname_dem_breached}",
                f"--dist={int(1000)}",
            ]
            if verbose:
                cmd.append("--verbose")
            result = subprocess.run(
                cmd,
                check=True,
                timeout=900,
                text=True,
            )
        except subprocess.TimeoutExpired:
            if verbose: print(" breach least cost timeout reached (15 min)")
        except Exception as e:
            if verbose: print(f" WARNING breach least cost failed with message {e}")
        if not os.path.isfile(fname_dem_breached): 
            print(" WARNING using less preferable fill depressions algorithm")
            wbt.fill_depressions(
                dem=fname_filled, 
                output=fname_dem_breached, 
                fix_flats=True, 
            )
        os.remove(fname_filled)
    else:
        if verbose: print(f' found existing breached dem {fname_dem_breached}')

def set_flow_acc(**kwargs):
    fname_dem_breached = kwargs.get('fname_dem_breached', None)
    fname_facc_ncells  = kwargs.get('fname_facc_ncells',  None)
    fname_facc_sca     = kwargs.get('fname_facc_sca',     None)
    verbose            = kwargs.get('verbose',            False)
    overwrite          = kwargs.get('overwrite',          False)
    if verbose: print(f'calling set_flow_acc')
    if not os.path.isfile(fname_facc_ncells) or overwrite:
        if verbose: print(f' using whitebox to calculate flow accumulation (n cells), writing to {fname_facc_ncells}')
        wbt = whitebox.WhiteboxTools()
        wbt.d_inf_flow_accumulation(i        = fname_dem_breached,
                                    output   = fname_facc_ncells,
                                    out_type = 'cells',
                                    log      = False)
    else:
        if verbose: print(f' using existing flow accumulation (ncells) file {fname_facc_ncells}')
    if not os.path.isfile(fname_facc_sca) or overwrite:
        if verbose: print(f' using whitebox to calculate flow accumulation (sca), writing to {fname_facc_sca}')
        wbt = whitebox.WhiteboxTools()
        wbt.d_inf_flow_accumulation(i        = fname_dem_breached,
                                    output   = fname_facc_sca,
                                    out_type = 'sca',
                                    log      = False)

    else:
        if verbose: print(f' using existing flow accumulation (sca) file {fname_facc_sca}')

def calc_stream_mask(**kwargs):
    verbose               = kwargs.get('verbose',              False)
    overwrite             = kwargs.get('overwrite',            False)
    fname_facc_ncells     = kwargs.get('fname_facc_ncells',    None)
    fname_facc_sca        = kwargs.get('fname_facc_sca',       None)
    facc_threshold_ncells = kwargs.get('facc_threshold_ncells',None)
    facc_threshold_sca    = kwargs.get('facc_threshold_sca',   None)
    fname_strm_mask       = kwargs.get('fname_strm_mask',      None)
    if verbose: print('calling calc_stream_mask')
    if not os.path.isfile(fname_strm_mask) or overwrite:
        if os.path.isfile(fname_facc_ncells):
            if verbose: print(f' setting stream mask using fname_facc_ncells {fname_facc_ncells} and facc_threshold_ncells {facc_threshold_ncells}')
            wbt = whitebox.WhiteboxTools()
            wbt.extract_streams(flow_accum      = fname_facc_ncells,
                                output          = fname_strm_mask,
                                threshold       = facc_threshold_ncells,
                                zero_background = True)
        elif os.path.isfile(fname_facc_sca):
            if verbose: print(f' setting stream mask using fname_facc_sca {fname_facc_sca} and facc_threshold_sca {facc_threshold_sca}')
            wbt = whitebox.WhiteboxTools()
            wbt.extract_streams(flow_accum      = fname_facc_sca,
                                output          = fname_strm_mask,
                                threshold       = facc_threshold_sca,
                                zero_background = True)
        else:
            raise Exception(f'calc_stream_mask did not find valid flow accumulation file fname_facc_ncells {fname_facc_ncells} or fname_facc_sca {fname_facc_sca}')
    else:
        if verbose: print(f' using existing stream mask {fname_strm_mask}')

def calc_slope(**kwargs):
    fname_dem_breached = kwargs.get('fname_dem_breached', None)
    fname_slope        = kwargs.get('fname_slope',        None)
    verbose            = kwargs.get('verbose',            False)
    overwrite          = kwargs.get('overwrite',          False)
    if verbose: print(f'calling calc_slope')
    if not os.path.isfile(fname_slope) or overwrite:
        wbt = whitebox.WhiteboxTools()
        wbt.slope(dem    = fname_dem_breached,
                  output = fname_slope,
                  units  = 'degrees')
    else:
        if verbose: print(f' using existing slope file {fname_slope}')

def calc_twi(**kwargs):
    fname_facc_sca     = kwargs.get('fname_facc_sca', None)
    fname_slope        = kwargs.get('fname_slope',    None)
    fname_twi          = kwargs.get('fname_twi',      None)
    verbose            = kwargs.get('verbose',        False)
    overwrite          = kwargs.get('overwrite',      False)
    if verbose: print('calling calc_twi')
    if not os.path.isfile(fname_twi) or overwrite:
        if verbose: print(f' using whitebox workflows to calculate {fname_twi}')
        wbt = whitebox.WhiteboxTools()
        wbt.wetness_index(sca    = fname_facc_sca,
                          slope  = fname_slope,
                          output = fname_twi)
    else:
        if verbose: print(f' found existing TWI file {fname_twi}')

def calc_twi_mean(**kwargs):
    """
    Compute the mean TWI per WTD grid cell over the full TWI extent,
    then reproject the result back to the original TWI grid so all TWI cells
    are present in the output (no cut-off).

    Required kwargs:
      - fname_twi: path to input TWI raster
      - fname_twi_mean: path to output raster
      - wtd_raw_dir: directory containing one or more WTD rasters (.tif/.tiff)

    Optional kwargs:
      - verbose: bool
      - overwrite: bool
      - compress: str (e.g., 'deflate' or 'lzw')
    """
    import os
    import glob
    import math
    import numpy as np
    import rioxarray
    import rasterio
    from affine import Affine
    from rasterio.warp import transform_bounds

    fname_twi          = kwargs.get('fname_twi',      None)
    fname_twi_mean     = kwargs.get('fname_twi_mean', None)
    wtd_raw_dir        = kwargs.get('wtd_raw_dir',    None)
    verbose            = kwargs.get('verbose',        False)
    overwrite          = kwargs.get('overwrite',      False)
    compress           = kwargs.get('compress',       'deflate')  # 'deflate' or 'lzw'

    if verbose:
        print('calling calc_twi_mean')

    if not fname_twi or not fname_twi_mean or not wtd_raw_dir:
        raise ValueError('fname_twi, fname_twi_mean, and wtd_raw_dir are required')

    if os.path.isfile(fname_twi_mean) and not overwrite:
        if verbose:
            print(f' found existing mean TWI file {fname_twi_mean}')
        return

    # Pick a representative WTD raster to define CRS/resolution/grid alignment
    wtd_candidates = sorted(
        glob.glob(os.path.join(wtd_raw_dir, '*.tif')) +
        glob.glob(os.path.join(wtd_raw_dir, '*.tiff'))
    )
    if not wtd_candidates:
        raise ValueError(f'calc_twi_mean could not locate any .tif/.tiff files in {wtd_raw_dir}')
    fname_example_wtd_raw = wtd_candidates[0]

    if verbose:
        print(f' calculating mean TWI per WTD cell and saving on TWI grid -> {fname_twi_mean}')

    with rioxarray.open_rasterio(fname_twi, masked=True) as riox_ds_twi, \
         rioxarray.open_rasterio(fname_example_wtd_raw, masked=True) as riox_ds_wtd:

        # Validate CRS presence
        if riox_ds_twi.rio.crs is None:
            raise ValueError('Input TWI raster has no CRS defined.')
        if riox_ds_wtd.rio.crs is None:
            raise ValueError('Example WTD raster has no CRS defined.')

        # Build a WTD-grid template that fully covers the TWI extent (prevents cut-off)
        wtd_crs = riox_ds_wtd.rio.crs
        wtd_transform = riox_ds_wtd.rio.transform()

        left_twi, bottom_twi, right_twi, top_twi = riox_ds_twi.rio.bounds()
        twi_crs = riox_ds_twi.rio.crs

        left_wtd, bottom_wtd, right_wtd, top_wtd = transform_bounds(
            twi_crs, wtd_crs, left_twi, bottom_twi, right_twi, top_twi, densify_pts=21
        )

        # Convert those bounds to WTD pixel indices and snap to grid
        inv = ~wtd_transform
        c0, r0 = inv * (left_wtd,  top_wtd)
        c1, r1 = inv * (right_wtd, bottom_wtd)

        col_min = math.floor(min(c0, c1))
        row_min = math.floor(min(r0, r1))
        col_max = math.ceil(max(c0, c1))
        row_max = math.ceil(max(r0, r1))

        width = int(col_max - col_min)
        height = int(row_max - row_min)

        if width <= 0 or height <= 0:
            raise ValueError('Computed WTD template has non-positive dimensions. Check input extents/CRS.')

        template_transform = wtd_transform * Affine.translation(col_min, row_min)

        # Aggregate TWI -> WTD grid using average resampling on the full-coverage template.
        # IMPORTANT: Do not pass src_nodata/dst_nodata; rioxarray manages those internally.
        twi_on_wtd = riox_ds_twi.rio.reproject(
            dst_crs=wtd_crs,
            transform=template_transform,
            shape=(height, width),
            resampling=rasterio.enums.Resampling.average
        )

        # Always project back to the TWI grid so the output has exactly the TWI footprint
        twi_mean = twi_on_wtd.rio.reproject_match(
            riox_ds_twi,
            resampling=rasterio.enums.Resampling.nearest
        )

        # Ensure floating dtype for averaged values
        if np.issubdtype(twi_mean.dtype, np.integer):
            twi_mean = twi_mean.astype('float32')
        else:
            # Keep float32 to reduce file size if float64
            try:
                twi_mean = twi_mean.astype('float32')
            except Exception:
                pass

        # Write output
        os.makedirs(os.path.dirname(fname_twi_mean) or '.', exist_ok=True)
        twi_mean.rio.to_raster(
            fname_twi_mean,
            compress=compress,
            tiled=True
        )

        if verbose:
            print(f' wrote {fname_twi_mean}')

def set_domain_mask(**kwargs):
    domain            = kwargs.get('domain',            None)
    verbose           = kwargs.get('verbose',           False)
    overwrite         = kwargs.get('overwrite',         False)
    fname_domain_mask = kwargs.get('fname_domain_mask', None)
    fname_dem         = kwargs.get('fname_dem',         None)
    if verbose: print(f'calling set_domain_mask')
    if not os.path.isfile(fname_dem):
        raise ValueError(f'set_domain_mask could not find fname_dem {fname_dem}')
    if not os.path.isfile(fname_domain_mask) or overwrite:
        if verbose: print(f' creating {fname_domain_mask}')
        with rioxarray.open_rasterio(fname_dem) as riox_ds_dem:
            domain = domain.to_crs(riox_ds_dem.rio.crs)
            domain = domain.dissolve()
            domain = domain.drop(columns=[col for col in domain.columns if col not in ['geometry']])
            domain['mask'] = 1
            domain_mask = make_geocube(vector_data=domain,like=riox_ds_dem,measurements=['mask'])
            domain_mask.rio.to_raster(fname_domain_mask)
    else:
        if verbose: print(f' using existing domain mask {fname_domain_mask}')






