import os
import sys
import asyncio
import twtnamelist
import twtdomain
import twtwt
import twttopo
import twtsoils
import twtstreams
import twtcalc
import datetime
import hf_hydrodata

async def calculate(fname_namelist):
    #
    #
    fname_namelist = os.path.abspath(str(fname_namelist))
    #
    #
    namelist = twtnamelist.Namelist(filename=fname_namelist)
    #
    #
    #if namelist.options.usedask:
    #chunks = {"y": 2048, "x": 2048}
    #if namelist.options.verbose:
    #    print(f'using dask with chunks {chunks}')
    #
    #
    kwargs = {'fname_domain' : namelist.fnames.domain,
              'verbose'      : namelist.options.verbose,
              'overwrite'    : namelist.options.overwrite,
              'conus1_domain': namelist.fnames.conus1_domain}
    if namelist.options.domain_hucid is not None:
        kwargs.update({'domain_hucid'     : namelist.options.domain_hucid})
    elif namelist.options.domain_latlon is not None:
        kwargs.update({'domain_latlon' : namelist.options.domain_latlon})
    elif namelist.options.domain_bbox is not None:
        kwargs.update({'domain_bbox'   : namelist.options.domain_bbox})
    else:
        if not os.path.isfile(namelist.fnames.domain):
            raise ValueError(f'calculate could not set domain from namelist options, and domain file {namelist.fnames.domain} does not exist')
    domain = twtdomain.set_domain(**kwargs)
    #
    #
    kwargs = {'domain'           : domain,
              'fname_domain_buf' : namelist.fnames.domain_buf,
              'buf_dist_m'       : namelist.options.domain_buf_dist_m,
              'verbose'          : namelist.options.verbose,
              'overwrite'        : namelist.options.overwrite}
    domain_buf = twtdomain.set_domain_buf(**kwargs)
    #
    #
    kwargs = {'dt_start'  : namelist.time.start_date,
              'dt_end'    : namelist.time.end_date,
              'dir_wtd'   : namelist.dirnames.wtd_raw,
              'verbose'   : namelist.options.verbose}
    wtd_get_flag = twtwt.set_wtd_get_flag(**kwargs)
    if namelist.options.verbose and not wtd_get_flag:
        print(f' found water table depth data for all dates in range in {namelist.dirnames.wtd_raw}')
    if wtd_get_flag and namelist.options.conus1_download_dir is None:
        kwargs = {'dt_start'  : namelist.time.start_date,
                  'dt_end'    : namelist.time.end_date,
                  'dir_wtd'   : namelist.dirnames.wtd_raw,
                  'domain'    : domain_buf,
                  'verbose'   : namelist.options.verbose,
                  'overwrite' : namelist.options.overwrite}
        hf_hydrodata.register_api_pin(namelist.options.hf_hydrodata_un, namelist.options.hf_hydrodata_pin)
        twtwt.download_hydroframe_data(**kwargs)
    elif wtd_get_flag and namelist.options.conus1_download_dir is not None:
        kwargs = {'dt_start'  : namelist.time.start_date,
                  'dt_end'    : namelist.time.end_date,
                  'wtd_in_dir': namelist.options.conus1_download_dir,
                  'wtd_out_dir': namelist.dirnames.wtd_raw,
                  'domain'    : domain_buf,
                  'verbose'   : namelist.options.verbose,
                  'overwrite' : namelist.options.overwrite}
        twtwt.break_conus1_tiffs(**kwargs)
    #
    #
    if namelist.fnames.dem_namelist_input is not None and os.path.isfile(namelist.fnames.dem_namelist_input):
        kwargs = {'fname_dem_parent' : namelist.fnames.dem_namelist_input,
                  'fname_dem_child'  : namelist.fnames.dem,
                  'fname_boundary'   : namelist.fnames.domain,
                  'verbose'          : namelist.options.verbose,
                  'overwrite'        : namelist.options.overwrite}
        twttopo.break_dem(**kwargs)
    else:
        kwargs = {'domain'    : domain,
                'dem_rez'   : namelist.options.dem_rez,
                'fname_dem' : namelist.fnames.dem,
                'verbose'   : namelist.options.verbose,
                'overwrite' : namelist.options.overwrite}
        await twttopo.download_dem(**kwargs)
    #
    #
    kwargs = {'fname_dem_breached' : namelist.fnames.dem_breached,
              'fname_dem'          : namelist.fnames.dem,
              'verbose'            : namelist.options.verbose,
              'overwrite'          : namelist.options.overwrite}
    twttopo.breach_dem(**kwargs)
    #
    #
    kwargs = {'fname_dem_breached' : namelist.fnames.dem_breached,
              'fname_facc_ncells'  : namelist.fnames.facc_ncells,
              'fname_facc_sca'     : namelist.fnames.facc_sca,
              'verbose'            : namelist.options.verbose,
              'overwrite'          : namelist.options.overwrite}
    twttopo.set_flow_acc(**kwargs)
    #
    #
    kwargs = {'fname_facc_ncells'     : namelist.fnames.facc_ncells,
              'facc_threshold_ncells' : namelist.options.facc_strm_thresh_ncells,
              'fname_strm_mask'       : namelist.fnames.stream_mask,
              'verbose'               : namelist.options.verbose,
              'overwrite'             : namelist.options.overwrite}
    twttopo.calc_stream_mask(**kwargs)
    #
    #
    kwargs = {'fname_dem_breached' : namelist.fnames.dem_breached,
              'fname_slope'        : namelist.fnames.slope,
              'verbose'            : namelist.options.verbose,
              'overwrite'          : namelist.options.overwrite}
    twttopo.calc_slope(**kwargs)
    #
    #
    kwargs = {'fname_facc_sca' : namelist.fnames.facc_sca,
              'fname_twi'      : namelist.fnames.twi,
              'fname_slope'    : namelist.fnames.slope,
              'verbose'        : namelist.options.verbose,
              'overwrite'      : namelist.options.overwrite}
    twttopo.calc_twi(**kwargs)
    #
    #
    kwargs = {'fname_twi_mean' : namelist.fnames.twi_mean,
              'fname_twi'      : namelist.fnames.twi,
              'wtd_raw_dir'    : namelist.dirnames.wtd_raw,
              'verbose'        : namelist.options.verbose,
              'overwrite'      : namelist.options.overwrite}
    twttopo.calc_twi_mean(**kwargs)
    #
    #
    if namelist.fnames.soil_texture_namelist_input is not None and os.path.isfile(namelist.fnames.soil_texture_namelist_input):
        kwargs = {'fname_texture_parent' : namelist.fnames.soil_texture_namelist_input,
                'fname_texture_child'    : namelist.fnames.soil_texture,
                'fname_domain'           : namelist.fnames.domain,
                'verbose'                : namelist.options.verbose,
                'overwrite'              : namelist.options.overwrite}
        twtsoils.break_soil_texture(**kwargs)
    else:
        kwargs = {'fname_texture'  : namelist.fnames.soil_texture,
                'domain'         : domain,
                'domain_buf'     : domain_buf,
                'verbose'        : namelist.options.verbose,
                'overwrite'      : namelist.options.overwrite}
        await twtsoils.download_soil_texture(**kwargs)
    #
    #
    kwargs = {'fname_texture'        : namelist.fnames.soil_texture,
              'fname_transmissivity' : namelist.fnames.soil_transmissivity,
              'fname_dem'            : namelist.fnames.dem_breached,
              'verbose'              : namelist.options.verbose,
              'overwrite'            : namelist.options.overwrite}
    twtsoils.set_soil_transmissivity(**kwargs)
    #
    #
    kwargs = {'fname_streams' : namelist.fnames.nhdp,
              'domain'        : domain,
              'verbose'       : namelist.options.verbose,
              'overwrite'     : namelist.options.overwrite}
    #try:
    #    twtstreams.set_streams(**kwargs)
    #except Exception as e:
    #    print(f'WARNING: failed to get NHD stream lines with error {e}')
    #
    #
    kwargs = {'dt_start'                  : namelist.time.start_date,
              'dt_end'                    : namelist.time.end_date,
              'wtd_raw_dir'               : namelist.dirnames.wtd_raw,
              'inundation_out_dir'        : namelist.dirnames.output_raw,
              'fname_soil_trans'          : namelist.fnames.soil_transmissivity,
              'fname_twi'                 : namelist.fnames.twi,
              'fname_twi_mean'            : namelist.fnames.twi_mean,
              'verbose'                   : namelist.options.verbose,
              'overwrite'                 : namelist.options.overwrite}
    twtcalc.calculate_inundation(**kwargs)
    #
    #
    kwargs = {'dt_start'                  : namelist.time.start_date,
              'dt_end'                    : namelist.time.end_date,
              'inundation_raw_dir'        : namelist.dirnames.output_raw,
              'inundation_summary_dir'    : namelist.dirnames.output_summary,
              'fname_dem'                 : namelist.fnames.dem_breached,
              'verbose'                   : namelist.options.verbose,
              'overwrite'                 : namelist.options.overwrite}
    fname_perc_inundated = twtcalc.calculate_summary_perc_inundated(**kwargs)
    #
    #
    kwargs = {'fname_perc_inundation'     : fname_perc_inundated,
              'fname_strm_mask'           : namelist.fnames.stream_mask,
              'verbose'                   : namelist.options.verbose,
              'overwrite'                 : namelist.options.overwrite}
    twtcalc.calculate_strm_permanence(**kwargs)
    #
    #
    return None

def calculate_async_wrapper(**kwargs):
    """async wrapper for calculation"""
    #
    #
    fname_namelist = kwargs.get('fname_namelist',None)
    if fname_namelist is None:
        raise KeyError('calculate_async_wrapper requires fname_namelist in kwargs')
    #
    #
    fname_verbose  = os.path.join(os.path.dirname(fname_namelist),
                                  f"verbose_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt")
    print(f'processing {fname_namelist}')
    f = open(fname_verbose, "a", buffering=1)
    sys.stdout = f
    sys.stderr = f
    #
    #
    try:
        asyncio.run(calculate(fname_namelist))
    except Exception as e:
        print(str(e))
    #
    #
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    f.close()
    #
    #
    return 