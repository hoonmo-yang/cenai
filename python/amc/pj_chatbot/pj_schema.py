from cenai_core.pandas_helper import (
    DataFrameSchema, schema_birthdate, schema_datetime, schema_integer
)

PJ_SCHEMA = {
    "patient":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 9)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "birth_ym",
                "sex_cd",
                "frst_vist_dt",
                "dx_dt",
                "prmr_orgn_cd",
                "mrph_diag_cd",
                "cancer_reg_dt",
                "death_dt",
            ],
            converters={
                "birth_ym": schema_birthdate,
                "frst_vist_dt": schema_datetime(False),
                "dx_dt": schema_datetime(False),
                "cancer_reg_dt": schema_datetime(False),
                "death_dt": schema_datetime(False),
            },
        ),
    "surgery":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 5)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "oprt_dt",
                "opdp_nm",
                "oprt_cd",
                "oprt_nm",
            ],
            converters={
                "oprt_dt": schema_datetime(False),
            },
        ),
    "tissue":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 5)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "tssu_exam_dt",
                "tssu_exam_cd",
                "tssu_exam_nm",
                "tssu_exam_rslt_text_cnte",
            ],
            converters={
                "tssu_exam_dt": schema_datetime(False),
            },
        ),
    "blbm":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 5)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "blbm_exam_dt",
                "blbm_exam_cd",
                "blbm_exam_eng_nm",
                "blbm_exam_rslt_text_cnte",
            ],
            converters={
                "blbm_exam_dt": schema_datetime(False),
            },
        ),
    "biopsy":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 6)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "bx_exam_dt",
                "bx_orgn_nm",
                "bx_exam_cd",
                "bx_exam_nm",
                "bx_exam_rslt_text_cnte",
            ],
            converters={
                "bx_exam_dt": schema_datetime(False),
            },
        ),
    "ihc":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 6)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "paex_div_cd",
                "paex_ihc_dt",
                "paex_ihc_cd",
                "paex_ihc_nm",
                "paex_ihc_rslt_ct_ctn",
            ],
            converters={
                "paex_ihc_dt": schema_datetime(False),
            },
        ),
    "sequencing":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 6)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "paex_sqnc",
                "paex_sqnc_dt",
                "paex_sqnc_cd",
                "paex_sqnc_nm",
                "paex_sqnc_rslt_ct_ctn",
            ],
            converters={
                "paex_sqnc_dt": schema_datetime(False),
            },
        ),
    "ct":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 5)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "imgx_ct_dt",
                "imgx_ct_cd",
                "imgx_ct_nm",
                "imgx_rslt_ct_ctn",
            ],
            converters={
                "imgx_ct_dt": schema_datetime(False),
            },
        ),
    "mr":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 5)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "imgx_mri_dt",
                "imgx_mri_cd",
                "imgx_mri_nm",
                "imgx_rslt_mri_ctn",
            ],
            converters={
                "imgx_mri_dt": schema_datetime(False),
            },
        ),
    "petct":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 5)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "imgx_petct_dt",
                "imgx_petct_cd",
                "imgx_petct_nm",
                "imgx_rslt_petct_ctn",
            ],
            converters={
                "imgx_petct_dt": schema_datetime(False),
            },
        ),
    "us":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 5)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "imgx_us_dt",
                "imgx_us_cd",
                "imgx_us_nm",
                "imgx_rslt_us_ctn",
            ],
            converters={
                "imgx_us_dt": schema_datetime(False),
            },
        ),
    "ercp":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 5)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "ercp_act_dt",
                "ercp_act_cd",
                "ercp_act_nm",
                "ercp_act_rslt_ctn",
            ],
            converters={
                "ercp_act_dt": schema_datetime(False),
            },
        ),
    "eus":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 5)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "eus_act_dt",
                "eus_act_cd",
                "eus_act_nm",
                "eus_act_rslt_ctn",
            ],
            converters={
                "eus_act_dt": schema_datetime(False),
            },
        ),
    "ctx":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 7)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "phrm_cd",
                "inhosp_medi_eng_nm",
                "medi_ingre_nm",
                "ordr_strt_dt",
                "ordr_end_dt",
                "ordr_days",
            ],
            converters={
                "ordr_strt_dt": schema_datetime(False),
                "ordr_end_dt": schema_datetime(False),
                "ordr_days": schema_integer,
            },
        ),
    "rtx":
        DataFrameSchema(
            serializer="xlsx",
            usecols=[range(0, 7)],
            skiprows=[1],
            columns=[
                "resch_pat_id",
                "rtx_nm",
                "daily_tret_lorr",
                "tot_tret_cnt",
                "tot_tret_lorr",
                "rad_tret_st_dt",
                "rad_tret_end_dt",
            ],
            converters={
                "daily_tret_lorr": schema_integer,
                "tot_tret_cnt": schema_integer,
                "tot_tret_lorr": schema_integer,
                "rad_tret_st_dt": schema_datetime(False),
                "rad_tret_end_dt": schema_datetime(False),
            },
        ),
}