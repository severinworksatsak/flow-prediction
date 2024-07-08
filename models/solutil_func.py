# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:49:48 2024

@author: LES
"""
from datetime import datetime, timedelta, date, timezone
from pytz import timezone
import pandas as pd
import os
from dotenv import load_dotenv
import oracledb
import warnings
from sklearn import metrics
import seaborn as sns


# Env Variable Retrieval
def get_env_variables(mandant:str, scope:str='get_ts'):
    """
    Retrieve setup variables from .env environment file, e.g., BelVis username & password.

    Parameters:
        - **mandant** (str): Any existing Belvis Mandant. Commonly used: 'EPAG_PFM' or 'EPAG_ENERGIE'.
        - **scope** (str): Scope of to-be-retrieved variables. Can be 'all' or 'get_ts'.

    Returns:
        - **env_dict** (dict): Dictionary with environment variables of the selected scope.

    Notes:
        With 'get_ts', the resulting env_dict can directly be passed to the get_timeseries
        function and unpacked with the ** operator.
    """
    # Try to load environment file if it exists
    if not load_dotenv():
        raise FileNotFoundError("No .env file exists in the current directory.")

    match scope:
        case 'all':
            env_dict = {key:value for (key, value) in os.environ}
        case 'get_ts':
            # Get input variables for get_timeseries functions
            env_keys = ['mandant_user', 'mandant_pwd', 'mandant_addr']
            mandant_user = os.getenv(f'{mandant}_USER')
            mandant_pwd = os.getenv(f'{mandant}_PWD')
            mandant_addr = os.getenv('BELVIS_ADDR_NEW')
            env_values = [mandant_user, mandant_pwd, mandant_addr]

            # Combine to dict
            env_dict = dict(zip(env_keys, env_values))
        case _:
            env_dict = None
            print(f'Scope input {scope} does not match options: [all, get_ts]')

    return env_dict


## 15min Time Series Retrieval
def get_timeseries_15min(ts_id: int, date_from, date_to,
                         mandant_user:str, mandant_pwd, mandant_addr:str,
                         offset_summertime: bool = False, col_name: str = 'value',
                         USE_OLD_CX_ORACLE:bool=False):
    """
    Extract 15-minute resolution time series from various BelVis database Mandanten.

    Parameters:
        - **ts_id** (int): The BelVis time series ID in integer format, e.g., 404212442.
        - **date_from** (dt): A datetime object representing the start date of the export window.
                          Start date is inclusive of `date_from` value, i.e., the start date will
                          be included in the extraction range. Format examples:
                          - datetime(2024, 1, 1, 0, 0, 0)
                          - datetime.strptime(date_string, format)
                          - any other datetime object
        - **date_to** (dt): A datetime object representing the end date of the export window.
                            The end date is not included in the queried output.
        - **mandant_user** (str): Username for the BelVis Mandant.
        - **mandant_pwd** (str): Password for the selected BelVis Mandant.
        - **mandant_addr"" (str): Address of the selected BelVis Mandant.
        - **offset_summertime** (bool): Indicator whether summertime is offset, i.e., whether
                                        timestamps are in Etc/GMT-1 or local time. If False, code will
                                        directly localize timestamp at Etc/GMT-1, else it will
                                        first convert to CET before Etc/GMT-1.
        - **col_name** (str): Column name which the output dataframe column should bear.
        - **USE_OLD_CX_ORACLE** (bool): Switch variable to use old database connection module. Deprecated under
                                        Python versions >3.6.

    Returns:
        - **data_out** (pd.Series): Dataframe with output time series; timestamp index and `col_name` name.
"""
    if date_from.tzinfo is None:
        # Convert start and end dates if summertime offset is desired
        if offset_summertime:
            date_from = timezone('CET').localize(date_from)
            date_to = timezone('CET').localize(date_to)
            date_from = date_from.astimezone(timezone('Etc/GMT-1'))
            date_to = date_to.astimezone(timezone('Etc/GMT-1')) # Etc/GMT-1 actually refers to GMT+1, as highlighted
            # in https://stackoverflow.com/questions/53076575/time-zones-etc-gmt-why-it-is-other-way-round
        else:
            # Localize timestamps to constant Etc/GMT-1
            date_from = timezone('Etc/GMT-1').localize(date_from)
            date_to = timezone('Etc/GMT-1').localize(date_to)

    # Convert back to str format for SQL Query
    str_from = date_from.strftime("%d.%m.%Y %H:%M:%S")
    str_to = date_to.strftime("%d.%m.%Y %H:%M:%S")

    # Define timedelta check query string
    check_str_sql = f"""select tsd.tstamp_ts,
                            tsd.tstamp_ts - lag(tsd.tstamp_ts) over (order by tsd.tstamp_ts) as date_diff
                        from tsd_wnmin195 tsd
                        where tsd.timeseries_l = (
                            select ident
                            from ts_timeseries ts
                            where ts.valuelist_l = {ts_id}
                            ) and
                            tsd.tstamp_ts >= to_timestamp('{str_from}', 'DD.MM.YYYY HH24:MI:SS') and
                            tsd.tstamp_ts < to_timestamp('{str_to}', 'DD.MM.YYYY HH24:MI:SS') and
                            ROWNUM <= 2
                    """

    # Define query string
    str_sql = f"""select zeitstempel, value from
                    (select
                        case when zeitstempel is not null
                                and zeitstempel >= next_day( to_date(concat( extract(year from zeitstempel),
                                                                                '0401.02'), 'yyyymmdd.HH') -8,7)
                                and zeitstempel < next_day( to_date(concat( extract(year from zeitstempel),
                                                                                '1101.03'), 'yyyymmdd.HH') -8,7)
                                then
                            zeitstempel + 0/24
                        else zeitstempel  end as zeitstempel, value, timeseries_l
                        from
                    (
                    select timeseries_l, tstamp_ts + 1/96 - 2/96 as zeitstempel, value1 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 2/96 - 2/96 as zeitstempel, value2 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 3/96 - 2/96 as zeitstempel, value3 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 4/96 - 2/96 as zeitstempel, value4 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 5/96 - 2/96 as zeitstempel, value5 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 6/96 - 2/96 as zeitstempel, value6 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 7/96 - 2/96 as zeitstempel, value7 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 8/96 - 2/96 as zeitstempel, value8 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 9/96 - 2/96 as zeitstempel, value9 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 10/96 - 2/96 as zeitstempel, value10 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 11/96 - 2/96 as zeitstempel, value11 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 12/96 - 2/96 as zeitstempel, value12 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 13/96 - 2/96 as zeitstempel, value13 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 14/96 - 2/96 as zeitstempel, value14 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 15/96 - 2/96 as zeitstempel, value15 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 16/96 - 2/96 as zeitstempel, value16 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 17/96 - 2/96 as zeitstempel, value17 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 18/96 - 2/96 as zeitstempel, value18 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 19/96 - 2/96 as zeitstempel, value19 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 20/96 - 2/96 as zeitstempel, value20 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 21/96 - 2/96 as zeitstempel, value21 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 22/96 - 2/96 as zeitstempel, value22 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 23/96 - 2/96 as zeitstempel, value23 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 24/96 - 2/96 as zeitstempel, value24 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 25/96 - 2/96 as zeitstempel, value25 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 26/96 - 2/96 as zeitstempel, value26 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 27/96 - 2/96 as zeitstempel, value27 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 28/96 - 2/96 as zeitstempel, value28 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 29/96 - 2/96 as zeitstempel, value29 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 30/96 - 2/96 as zeitstempel, value30 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 31/96 - 2/96 as zeitstempel, value31 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 32/96 - 2/96 as zeitstempel, value32 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 33/96 - 2/96 as zeitstempel, value33 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 34/96 - 2/96 as zeitstempel, value34 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 35/96 - 2/96 as zeitstempel, value35 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 36/96 - 2/96 as zeitstempel, value36 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 37/96 - 2/96 as zeitstempel, value37 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 38/96 - 2/96 as zeitstempel, value38 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 39/96 - 2/96 as zeitstempel, value39 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 40/96 - 2/96 as zeitstempel, value40 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 41/96 - 2/96 as zeitstempel, value41 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 42/96 - 2/96 as zeitstempel, value42 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 43/96 - 2/96 as zeitstempel, value43 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 44/96 - 2/96 as zeitstempel, value44 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 45/96 - 2/96 as zeitstempel, value45 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 46/96 - 2/96 as zeitstempel, value46 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 47/96 - 2/96 as zeitstempel, value47 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 48/96 - 2/96 as zeitstempel, value48 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 49/96 - 2/96 as zeitstempel, value49 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 50/96 - 2/96 as zeitstempel, value50 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 51/96 - 2/96 as zeitstempel, value51 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 52/96 - 2/96 as zeitstempel, value52 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 53/96 - 2/96 as zeitstempel, value53 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 54/96 - 2/96 as zeitstempel, value54 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 55/96 - 2/96 as zeitstempel, value55 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 56/96 - 2/96 as zeitstempel, value56 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 57/96 - 2/96 as zeitstempel, value57 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 58/96 - 2/96 as zeitstempel, value58 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 59/96 - 2/96 as zeitstempel, value59 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 60/96 - 2/96 as zeitstempel, value60 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 61/96 - 2/96 as zeitstempel, value61 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 62/96 - 2/96 as zeitstempel, value62 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 63/96 - 2/96 as zeitstempel, value63 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 64/96 - 2/96 as zeitstempel, value64 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 65/96 - 2/96 as zeitstempel, value65 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 66/96 - 2/96 as zeitstempel, value66 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 67/96 - 2/96 as zeitstempel, value67 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 68/96 - 2/96 as zeitstempel, value68 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 69/96 - 2/96 as zeitstempel, value69 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 70/96 - 2/96 as zeitstempel, value70 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 71/96 - 2/96 as zeitstempel, value71 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 72/96 - 2/96 as zeitstempel, value72 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 73/96 - 2/96 as zeitstempel, value73 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 74/96 - 2/96 as zeitstempel, value74 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 75/96 - 2/96 as zeitstempel, value75 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 76/96 - 2/96 as zeitstempel, value76 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 77/96 - 2/96 as zeitstempel, value77 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 78/96 - 2/96 as zeitstempel, value78 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 79/96 - 2/96 as zeitstempel, value79 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 80/96 - 2/96 as zeitstempel, value80 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 81/96 - 2/96 as zeitstempel, value81 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 82/96 - 2/96 as zeitstempel, value82 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 83/96 - 2/96 as zeitstempel, value83 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 84/96 - 2/96 as zeitstempel, value84 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 85/96 - 2/96 as zeitstempel, value85 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 86/96 - 2/96 as zeitstempel, value86 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 87/96 - 2/96 as zeitstempel, value87 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 88/96 - 2/96 as zeitstempel, value88 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 89/96 - 2/96 as zeitstempel, value89 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 90/96 - 2/96 as zeitstempel, value90 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 91/96 - 2/96 as zeitstempel, value91 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 92/96 - 2/96 as zeitstempel, value92 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 93/96 - 2/96 as zeitstempel, value93 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 94/96 - 2/96 as zeitstempel, value94 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 95/96 - 2/96 as zeitstempel, value95 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 96/96 - 2/96 as zeitstempel, value96 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) ))
                where zeitstempel >= to_timestamp('{str_from}', 'DD.MM.YYYY HH24:MI:SS') and
                        zeitstempel < to_timestamp('{str_to}', 'DD.MM.YYYY HH24:MI:SS')"""
    # logging.info("before db data fetching")

    # DB Connection and Data Retrieval
    if USE_OLD_CX_ORACLE:
        with cx_Oracle.connect(mandant_user, mandant_pwd, mandant_addr) as db_conn: # xc_Oracle only seems to run with Python versions 3.6 thru 3.10
            db_cursor = db_conn.cursor()
            ts_cursor = db_cursor.execute(str_sql).fetchall()
    else:
        # New db connection module
        with oracledb.connect(user=mandant_user, password=mandant_pwd,
                              dsn=mandant_addr) as db_conn:
            db_cursor = db_conn.cursor()

            # Check timedelta conformity with get_timeseries interval
            ts_check = db_cursor.execute(check_str_sql).fetchall()
            df_check = pd.DataFrame([row[1] for row in ts_check], index=[row[0] for row in ts_check], columns=['value'])
            if len(df_check) > 1:
                if df_check.iloc[1, 0] != 1:
                    print(warnings.warn(f"Warning: Time series with ID {ts_id} might not be in 15min resolution."))

            # Fetch data from db
            ts_cursor = db_cursor.execute(str_sql).fetchall()

    # Post-Query Data Handling
    data_db = pd.DataFrame([row[1] for row in ts_cursor], index=[row[0] for row in ts_cursor], columns=['value'])
    if len(data_db) > 0:
        data_db.index = data_db.index.tz_localize('Etc/GMT-1')

    # Reindexing & Assignment of Return Variable
    index_ts = pd.date_range(start=date_from, end=date_to - timedelta(minutes=15), freq='15min')
    data_out = pd.DataFrame(index=index_ts, columns=[col_name])
    data_out.index.name = 'Timestamp'
    if len(data_db) > 0:
        data_out['value'] = data_db['value']
    if offset_summertime:
        data_out.index = data_out.index.tz_convert('CET')
    data_out['value'] = pd.to_numeric(data_out['value'])

    # Convert to pd.Series if only one column
    if len(data_out.columns) <= 1:
        data_out = data_out['value']

    return data_out


## 1h Time Series Retrieval
def get_timeseries_1h(ts_id: int, date_from, date_to,
                         mandant_user:str, mandant_pwd, mandant_addr:str,
                         offset_summertime: bool = False, col_name: str = 'value',
                         USE_OLD_CX_ORACLE:bool=False):
    """
    Extract 1-hour resolution time series from various BelVis database Mandanten.

    Parameters:
        - **ts_id** (int): The BelVis time series ID in integer format, e.g., 404212442.
        - **date_from** (dt): A datetime object representing the start date of the export window.
                              Start date is inclusive of `date_from` value, i.e., the start date will
                              be included in the extraction range. Format examples:
                              - datetime(2024, 1, 1, 0, 0, 0)
                              - datetime.strptime(date_string, format)
                              - any other datetime object
        - **date_to** (dt): A datetime object representing the end date of the export window.
                            The end date is not included in the queried output.
        - **mandant_user** (str): Username for the BelVis Mandant.
        - **mandant_pwd** (str): Password for the selected BelVis Mandant.
        - **mandant_addr"" (str): Address of the selected BelVis Mandant.
        - **offset_summertime** (bool): Indicator whether summertime is offset, i.e., whether
                                        timestamps are in CET or local time. If False, code will
                                        directly localize timestamp at Etc/GMT-1, else it will
                                        first convert to CET before Etc/GMT-1.
        - **col_name** (str): Column name which the output dataframe column should bear.
        - **USE_OLD_CX_ORACLE** (bool): Switch variable to use old database connection module. Deprecated under
                                        Python versions >3.6.

    Returns:
        - **data_out** (pd.Series): Dataframe with output time series; timestamp index and `col_name` name.
    """

    if date_from.tzinfo is None:
        if offset_summertime:
            date_from = timezone('CET').localize(date_from)
            date_to = timezone('CET').localize(date_to)
            date_from = date_from.astimezone(timezone('Etc/GMT-1'))
            date_to = date_to.astimezone(timezone('Etc/GMT-1'))
        else:
            date_from = timezone('Etc/GMT-1').localize(date_from)
            date_to = timezone('Etc/GMT-1').localize(date_to)

    # Convert back to str format for SQL Query
    str_from = date_from.strftime("%d.%m.%Y %H:%M:%S")
    str_to = date_to.strftime("%d.%m.%Y %H:%M:%S")

    # Define timedelta check query string
    check_str_sql = f"""select tsd.tstamp_ts,
                            tsd.tstamp_ts - lag(tsd.tstamp_ts) over (order by tsd.tstamp_ts) as date_diff
                        from tsd_wnmin195 tsd
                        where tsd.timeseries_l = (
                            select ident
                            from ts_timeseries ts
                            where ts.valuelist_l = {ts_id}
                            ) and
                            tsd.tstamp_ts >= to_timestamp('{str_from}', 'DD.MM.YYYY HH24:MI:SS') and
                            tsd.tstamp_ts < to_timestamp('{str_to}', 'DD.MM.YYYY HH24:MI:SS') and
                            ROWNUM <= 2
                    """

    # Define query string
    str_sql = f"""select zeitstempel, value from 
                    (select
                        case when zeitstempel is not null 
                                and zeitstempel >= next_day( to_date(concat( extract(year from zeitstempel), 
                                                                                '0401.02'), 'yyyymmdd.HH') -8,7)  
                                and zeitstempel < next_day( to_date(concat( extract(year from zeitstempel), 
                                                                                '1101.03'), 'yyyymmdd.HH') -8,7)
                                then 
                            zeitstempel + 0/24
                        else zeitstempel  end as zeitstempel, value, timeseries_l
                        from
                    (
                    select timeseries_l, tstamp_ts + 1/24 - 2/24 as zeitstempel, value1 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 2/24 - 2/24 as zeitstempel, value2 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 3/24 - 2/24 as zeitstempel, value3 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 4/24 - 2/24 as zeitstempel, value4 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 5/24 - 2/24 as zeitstempel, value5 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 6/24 - 2/24 as zeitstempel, value6 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 7/24 - 2/24 as zeitstempel, value7 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 8/24 - 2/24 as zeitstempel, value8 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 9/24 - 2/24 as zeitstempel, value9 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 10/24 - 2/24 as zeitstempel, value10 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 11/24 - 2/24 as zeitstempel, value11 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 12/24 - 2/24 as zeitstempel, value12 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 13/24 - 2/24 as zeitstempel, value13 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 14/24 - 2/24 as zeitstempel, value14 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 15/24 - 2/24 as zeitstempel, value15 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 16/24 - 2/24 as zeitstempel, value16 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 17/24 - 2/24 as zeitstempel, value17 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 18/24 - 2/24 as zeitstempel, value18 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 19/24 - 2/24 as zeitstempel, value19 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 20/24 - 2/24 as zeitstempel, value20 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 21/24 - 2/24 as zeitstempel, value21 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 22/24 - 2/24 as zeitstempel, value22 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 23/24 - 2/24 as zeitstempel, value23 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 24/24 - 2/24 as zeitstempel, value24 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 25/24 - 2/24 as zeitstempel, value25 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 26/24 - 2/24 as zeitstempel, value26 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 27/24 - 2/24 as zeitstempel, value27 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 28/24 - 2/24 as zeitstempel, value28 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 29/24 - 2/24 as zeitstempel, value29 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 30/24 - 2/24 as zeitstempel, value30 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 31/24 - 2/24 as zeitstempel, value31 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 32/24 - 2/24 as zeitstempel, value32 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 33/24 - 2/24 as zeitstempel, value33 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 34/24 - 2/24 as zeitstempel, value34 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 35/24 - 2/24 as zeitstempel, value35 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 36/24 - 2/24 as zeitstempel, value36 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 37/24 - 2/24 as zeitstempel, value37 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 38/24 - 2/24 as zeitstempel, value38 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 39/24 - 2/24 as zeitstempel, value39 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 40/24 - 2/24 as zeitstempel, value40 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 41/24 - 2/24 as zeitstempel, value41 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 42/24 - 2/24 as zeitstempel, value42 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 43/24 - 2/24 as zeitstempel, value43 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 44/24 - 2/24 as zeitstempel, value44 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 45/24 - 2/24 as zeitstempel, value45 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 46/24 - 2/24 as zeitstempel, value46 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 47/24 - 2/24 as zeitstempel, value47 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 48/24 - 2/24 as zeitstempel, value48 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 49/24 - 2/24 as zeitstempel, value49 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 50/24 - 2/24 as zeitstempel, value50 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 51/24 - 2/24 as zeitstempel, value51 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 52/24 - 2/24 as zeitstempel, value52 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 53/24 - 2/24 as zeitstempel, value53 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 54/24 - 2/24 as zeitstempel, value54 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 55/24 - 2/24 as zeitstempel, value55 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 56/24 - 2/24 as zeitstempel, value56 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 57/24 - 2/24 as zeitstempel, value57 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 58/24 - 2/24 as zeitstempel, value58 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 59/24 - 2/24 as zeitstempel, value59 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 60/24 - 2/24 as zeitstempel, value60 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 61/24 - 2/24 as zeitstempel, value61 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 62/24 - 2/24 as zeitstempel, value62 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 63/24 - 2/24 as zeitstempel, value63 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 64/24 - 2/24 as zeitstempel, value64 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 65/24 - 2/24 as zeitstempel, value65 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 66/24 - 2/24 as zeitstempel, value66 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 67/24 - 2/24 as zeitstempel, value67 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 68/24 - 2/24 as zeitstempel, value68 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 69/24 - 2/24 as zeitstempel, value69 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 70/24 - 2/24 as zeitstempel, value70 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 71/24 - 2/24 as zeitstempel, value71 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 72/24 - 2/24 as zeitstempel, value72 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 73/24 - 2/24 as zeitstempel, value73 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 74/24 - 2/24 as zeitstempel, value74 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 75/24 - 2/24 as zeitstempel, value75 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 76/24 - 2/24 as zeitstempel, value76 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 77/24 - 2/24 as zeitstempel, value77 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 78/24 - 2/24 as zeitstempel, value78 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 79/24 - 2/24 as zeitstempel, value79 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 80/24 - 2/24 as zeitstempel, value80 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 81/24 - 2/24 as zeitstempel, value81 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 82/24 - 2/24 as zeitstempel, value82 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 83/24 - 2/24 as zeitstempel, value83 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 84/24 - 2/24 as zeitstempel, value84 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 85/24 - 2/24 as zeitstempel, value85 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 86/24 - 2/24 as zeitstempel, value86 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 87/24 - 2/24 as zeitstempel, value87 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 88/24 - 2/24 as zeitstempel, value88 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 89/24 - 2/24 as zeitstempel, value89 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 90/24 - 2/24 as zeitstempel, value90 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 91/24 - 2/24 as zeitstempel, value91 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 92/24 - 2/24 as zeitstempel, value92 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 93/24 - 2/24 as zeitstempel, value93 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 94/24 - 2/24 as zeitstempel, value94 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 95/24 - 2/24 as zeitstempel, value95 as value from tsd_wnmin195 
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) union
                    select timeseries_l, tstamp_ts + 96/24 - 2/24 as zeitstempel, value96 as value from tsd_wnmin195
                    where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id}) ))
                where zeitstempel >= to_timestamp('{str_from}', 'DD.MM.YYYY HH24:MI:SS') and 
                        zeitstempel < to_timestamp('{str_to}', 'DD.MM.YYYY HH24:MI:SS')"""

    # Database Connection & Data Retrieval
    if USE_OLD_CX_ORACLE:
        with cx_Oracle.connect(mandant_user, mandant_pwd, mandant_addr) as db_conn: # xc_Oracle only seems to run with Python versions 3.6 thru 3.10
            db_cursor = db_conn.cursor()
            ts_cursor = db_cursor.execute(str_sql).fetchall()
    else:
        # New db connection module
        with oracledb.connect(user=mandant_user, password=mandant_pwd,
                              dsn=mandant_addr) as db_conn:
            db_cursor = db_conn.cursor()

            # Check timedelta conformity with get_timeseries interval
            ts_check = db_cursor.execute(check_str_sql).fetchall()
            df_check = pd.DataFrame([row[1] for row in ts_check], index=[row[0] for row in ts_check], columns=['value'])
            if len(df_check) > 1:
                if df_check.iloc[1, 0] != 4:
                    print(warnings.warn(f"Warning: Time series with ID {ts_id} might not be in 1h resolution."))

            ts_cursor = db_cursor.execute(str_sql).fetchall()

    # Post-Query Data Handling
    data_db = pd.DataFrame([row[1] for row in ts_cursor], index=[row[0] for row in ts_cursor], columns=['value'])
    if len(data_db) > 0:
        data_db.index = data_db.index.tz_localize('Etc/GMT-1')

    # Reindexing & Assignment of Return Variable
    index_ts = pd.date_range(start=date_from, end=date_to - timedelta(minutes=60), freq='1h')
    data_out = pd.DataFrame(index=index_ts, columns=[col_name])
    data_out.index.name = 'Timestamp'
    if len(data_db) > 0:
        data_out['value'] = data_db['value']
    if offset_summertime:
        data_out.index = data_out.index.tz_convert('CET')
    data_out['value'] = pd.to_numeric(data_out['value'])

    # Convert to pd.Series if only one column
    if len(data_out.columns) <= 1:
        data_out = data_out['value']

    return data_out


## 1d Time Series Retrieval
def get_timeseries_1d(ts_id:int, date_from, date_to,
                      mandant_user:str, mandant_pwd, mandant_addr:str,
                      col_name:str='value', str_table:str='meanvalues',
                      USE_OLD_CX_ORACLE:bool=False):
    """
    Extract 1-day resolution time series from various BelVis database Mandanten.

    Parameters:
        - **ts_id** (int): The BelVis time series ID in integer format, e.g., 404212442.
        - **date_from** (dt): A datetime object representing the start date of the export window.
                              Start date is inclusive of `date_from` value, i.e., the start date will
                              be included in the extraction range. Format examples:
                              - datetime(2024, 1, 1, 0, 0, 0)
                              - datetime.strptime(date_string, format)
                              - any other datetime object
        - **date_to** (dt): A datetime object representing the end date of the export window.
                            The end date is not included in the queried output.
        - **mandant_user** (str): Username for the BelVis Mandant.
        - **mandant_pwd** (str): Password for the selected BelVis Mandant.
        - **mandant_addr"" (str): Address of the selected BelVis Mandant.
        - **col_name** (str): Column name which the output dataframe column should bear.
        - **str_table** (str): Table name suffix for ts_{str_table} to determine target table as specified by
                               SQL database structure. Defaults to 'meanvalues', other option is 'sumvalues'.

    Returns:
        - **data_out** (pd.Series): Dataframe with output time series; timestamp index and `col_name` name.
    """

    # Convert datetime to string for SQL Input
    str_from = date_from.strftime("%d.%m.%Y")
    str_to = date_to.strftime("%d.%m.%Y")

    # Define timedelta check query string
    check_str_sql = f"""select tsd.tstamp_ts,
                            tsd.tstamp_ts - lag(tsd.tstamp_ts) over (order by tsd.tstamp_ts) as date_diff
                        from tsd_{str_table} tsd
                        where tsd.timeseries_l = (
                            select ident
                            from ts_timeseries ts
                            where ts.valuelist_l = {ts_id}
                            ) and
                            tsd.tstamp_ts >= to_date('{str_from}', 'DD.MM.YYYY') and
                            tsd.tstamp_ts < to_date('{str_to}', 'DD.MM.YYYY') and
                            ROWNUM <= 2
                    """

    # Define Query String
    str_sql = f"""select tstamp_ts, value0_fl, state_l 
                  from tsd_{str_table}
                  where timeseries_l = (select min(ident) from ts_timeseries where valuelist_l = {ts_id})
                    and tstamp_ts >= to_date('{str_from}', 'DD.MM.YYYY')
                    and tstamp_ts < to_date('{str_to}', 'DD.MM.YYYY')"""

    # Database Connection & Data Retrieval
    if USE_OLD_CX_ORACLE:
        with cx_Oracle.connect(mandant_user, mandant_pwd, mandant_addr) as db_conn: # xc_Oracle only seems to run with Python versions 3.6 thru 3.10
            db_cursor = db_conn.cursor()
            ts_cursor = db_cursor.execute(str_sql).fetchall()
    else:
        # New db connection module
        with oracledb.connect(user=mandant_user, password=mandant_pwd,
                              dsn=mandant_addr) as db_conn:
            db_cursor = db_conn.cursor()

            # Check timedelta conformity with get_timeseries interval
            ts_check = db_cursor.execute(check_str_sql).fetchall()
            df_check = pd.DataFrame([row[1] for row in ts_check], index=[row[0] for row in ts_check], columns=['value'])
            if len(df_check) > 1:
                if df_check.iloc[1, 0] != 1:
                    print(warnings.warn(f"Warning: Time series with ID {ts_id} might not be in 1d resolution."))

            ts_cursor = db_cursor.execute(str_sql).fetchall()

    # Post-Query Data Handling
    index_ts = pd.date_range(start=date_from, end=date_to - timedelta(days=1), freq='1d')
    data_out = pd.DataFrame(index=index_ts, columns=[col_name])
    data_out.index.name = 'Timestamp'

    # Exception Handling based on state_l
    for i in range(len(ts_cursor)):
        if ts_cursor[i][2] != 285212672:
            data_out.loc[ts_cursor[i][0], col_name] = ts_cursor[i][1]

    # Timezone & Data Type Conversion
    data_out.index = data_out.index.tz_localize('Etc/GMT-1')
    data_out[col_name] = pd.to_numeric(data_out[col_name]).copy()

    # Convert to Series
    if len(data_out.columns) <= 1:
        data_out = data_out[col_name]

    return data_out


# def get_ts_info(ts_id, mandant_user:str, mandant_pwd, mandant_addr:str):
#
#     # SQL Query String
#     str_sql = f"""
#         select vall.name_zr_s as ts_name,
#             vall.name_inst_s as ts_parent,
#             vall.ident_vl_l as ts_id,
#             vall.einheit_s as ts_unit,
#             vall.creation_ts as ts_creation,
#             vall.lastsave_ts as ts_lastsave
#         from v_zr_all vall
#         where vall.ident_vl_l = TO_NUMBER(:ts_int)
#     """
#     print(str_sql)
#     with oracledb.connect(user=mandant_user, password=mandant_pwd,
#                           dsn=mandant_addr) as db_conn:
#         db_cursor = db_conn.cursor()
#
#         # Fetch data from db
#         ts_cursor = db_cursor.execute(str_sql, {'ts_int': ts_id}).fetchall()
#
#     # Post-Query Data Handling
#     data_db = pd.DataFrame.from_records(ts_cursor, columns =['ts_name', 'ts_parent', 'ts_id', 'ts_unit',
#                                                              'ts_creation', 'ts_lastsave'])
#
#     return data_db


def ts_name_lookup(str_name:str, mandant_user:str, mandant_pwd, mandant_addr:str):
    """
    Find time series from a string pattern search of all available time series in a Belvis mandant.

    Parameters:
        - **str_name** (str): String pattern for 'WHERE LIKE {}' SQL clause. Example: '%NM%Prognose'
                              for time series 'Niederschlagsmenge.Prognose.Meteomatics'
        - **mandant_user** (str): Username for the BelVis Mandant.
        - **mandant_pwd** (str): Password for the selected BelVis Mandant.
        - **mandant_addr** (str): Address of the selected BelVis Mandant.

    Returns:
        - **pd.DataFrame**: Time series overview for ts names with matching patterns. Includes name,
                        ts ids, parent infos, creation and lastsave dates.
    """
    # SQL Query String
    str_sql = f"""
        select vall.name_zr_s as ts_name,
            vall.name_inst_s as ts_parent,
            vall.ident_vl_l as ts_id, 
            vall.einheit_s as ts_unit,
            vall.creation_ts as ts_creation,
            vall.lastsave_ts as ts_lastsave
        from v_zr_all vall
        where vall.name_zr_s like :name_pattern
    """

    with oracledb.connect(user=mandant_user, password=mandant_pwd,
                          dsn=mandant_addr) as db_conn:
        db_cursor = db_conn.cursor()

        # Fetch data from db
        ts_cursor = db_cursor.execute(str_sql, {'name_pattern': str_name}).fetchall()

    # Post-Query Data Handling
    data_db = pd.DataFrame.from_records(ts_cursor, columns =['ts_name', 'ts_parent','ts_id', 'ts_unit', 'ts_creation', 'ts_lastsave'])

    return data_db


# Evaluation metrics
def get_eval_metrics(y_true, y_pred):
    """
    Calculate Evaluation Metrics MAE, MSE, RMSE, and MAPE.

    Parameters:
        - **y_true** (pandas.Series): Actual observations of the target variable.
        - **y_pred** (pandas.Series): Predicted values of the target variable.

    Returns:
        - **metric_dict** (dict): Dict containing evaluation metrics as values.
    """
    if len(y_true) != len(y_pred):
        raise warnings.warn('Warning: Length of input series not identical.')

    df_comb = y_true.to_frame(name='y_true')
    df_comb['y_pred'] = y_pred
    df_comb.dropna(inplace=True)

    # Calculate Metrics
    mae = metrics.mean_absolute_error(df_comb['y_true'], df_comb['y_pred'])
    mse = metrics.mean_squared_error(df_comb['y_true'], df_comb['y_pred'])
    rmse = metrics.root_mean_squared_error(df_comb['y_true'], df_comb['y_pred'])
    mape = metrics.mean_absolute_percentage_error(df_comb['y_true'], df_comb['y_pred']) # changed from mae
    nonna = round(len(df_comb) / len(y_true), 2)
    len_true = len(y_true)
    nnas = len_true - len(df_comb)

    # Create Metric Dict
    keys = ['mae', 'mse', 'rmse', 'mape', 'non-na%', 'n_actual_obs', 'n_pred_nas']
    values = [mae, mse, rmse, mape, nonna, len_true, nnas]
    metric_dict = dict(zip(keys, values))

    return metric_dict


# Actual vs. Prediction Plot
def get_act_vs_pred_plot(y_true, y_pred):
    """
    Create actual vs. prediction line plot.

    Parameters:
        - **y_true** (pandas.Series): Actual observations of the target variable.
        - **y_pred** (pandas.Series): Predicted values of the target variable.

    Returns:
        - **fig** (matplotlib.figure.Figure): Figure object containing axes filled with line plot.
    """
    # Merge series
    y_true.name = 'actual'
    df_data = y_true.to_frame()
    df_data['predicted'] = y_pred
    df_data['time'] = df_data.index

    # Melt dataframe
    df_melt = pd.melt(df_data, id_vars='time')

    # Create figure
    fig = sns.lineplot(data=df_melt, x='time', y='value', hue='variable', palette=['#660066', '#FFEB00'])

    return fig



