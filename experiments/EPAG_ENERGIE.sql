select vall.name_zr_s as ts_name,
    vall.name_inst_s as ts_parent,
    vall.ident_vl_l as ts_id,
    vall.einheit_s as ts_unit,
    vall.creation_ts as ts_creation,
    vall.lastsave_ts as ts_lastsave
from v_zr_all vall
where vall.ident_vl_l in (
    11127586,
    11055778,
    11055632,
    10253447,
    10255110,
    10253455,
    11010900,
    11010892,
    11055610
)
;