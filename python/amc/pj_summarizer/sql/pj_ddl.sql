CREATE TABLE IF NOT EXISTS public.patient (
    patient_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    resch_pat_id int4 NOT NULL, -- 연구환자ID
    birth_ym date, -- 생년월
    sex_cd varchar, -- 성별코드
    frst_vist_dt timestamp, -- 최초내원일자
    dx_dt date, -- 진단일자
    prmr_orgn_cd varchar, -- 원발장기코드
    mrph_diag_cd varchar, -- 형태학적진단코드
    cancer_reg_dt timestamp, -- 암등록일
    death_dt timestamp, -- 사망일자

    CONSTRAINT patient_pk PRIMARY KEY (patient_id),
    CONSTRAINT patient_resch_pat_id_un UNIQUE (resch_pat_id)
);

CREATE TABLE IF NOT EXISTS public.surgery (
    surgery_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    oprt_dt timestamp NULL, -- 수술일자
    opdp_nm varchar NULL, -- 집도과
    oprt_cd varchar NULL, -- 수술코드
    oprt_nm varchar NULL, -- 수술명
    resch_pat_id int4 NULL,

    CONSTRAINT surgery_pk PRIMARY KEY (surgery_id),
    CONSTRAINT surgery_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.tissue (
    tissue_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    tssu_exam_dt timestamp NULL, -- Tissue검사일자
    tssu_exam_cd varchar NULL, -- Tissue검사코드
    tssu_exam_nm varchar NULL, -- Tissue검사명
    tssu_exam_rslt_text_cnte varchar NULL, -- Tissue검사결과내용
    resch_pat_id int4 NULL,

    CONSTRAINT tissue_pk PRIMARY KEY (tissue_id),
    CONSTRAINT tissue_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.blbm (
    blbm_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    blbm_exam_dt timestamp NULL, -- Blood&BoneMarrow검사일자
    blbm_exam_cd varchar NULL, -- Blood&BoneMarrow검사코드
    blbm_exam_eng_nm varchar NULL, -- Blood&BoneMarrow검사명
    blbm_exam_rslt_text_cnte varchar NULL, -- Bone&BoneMarrow검사결과내용
    resch_pat_id int4 NULL,

    CONSTRAINT blbm_pk PRIMARY KEY (blbm_id),
    CONSTRAINT blbm_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.biopsy (
    biopsy_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    bx_exam_dt timestamp NULL, -- Biopsy검사일자
    bx_orgn_nm varchar NULL, -- Biopsy검사기관명
    bx_exam_cd varchar NULL, -- Biopsy검사코드
    bx_exam_nm varchar NULL, -- Biopsy 검사명
    bx_exam_rslt_text_cnte varchar NULL, -- Biopsy검사결과내용
    resch_pat_id int4 NULL,

    CONSTRAINT biopsy_pk PRIMARY KEY (biopsy_id),
    CONSTRAINT biopsy_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.ihc (
    ihc_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    paex_div_cd varchar NULL, -- 병리검사IHC
    paex_ihc_dt timestamp NULL, -- 병리검사IHC검사일자
    paex_ihc_cd varchar NULL, -- 병리검사IHC검사코드
    paex_ihc_nm varchar NULL, -- 병리검사IHC검사명
    paex_ihc_rslt_ct_ctn varchar NULL, -- 병리검사IHCR검사결과내용
    resch_pat_id int4 NULL,

    CONSTRAINT ihc_pk PRIMARY KEY (ihc_id),
    CONSTRAINT ihc_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.sequencing (
    sequencing_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    paex_sqnc varchar NULL, -- 병리검사Sequencing 
    paex_sqnc_dt timestamp NULL, -- 병리검사Sequencing검사일자
    paex_sqnc_cd varchar NULL, -- 병리검사Sequecing검사코드
    paex_sqnc_nm varchar NULL, -- 병리검사Sequencing검사명
    paex_sqnc_rslt_ct_ctn varchar NULL, -- 병리검사Sequencing검사결과내용
    resch_pat_id int4 NULL,

    CONSTRAINT sequencing_pk PRIMARY KEY (sequencing_id),
    CONSTRAINT sequencing_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.ct (
    ct_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    imgx_ct_dt timestamp NULL, -- 영상검사CT검사일자
    imgx_ct_cd varchar NULL, -- 영상검사CT검사코드
    imgx_ct_nm varchar NULL, -- 영상검사CT검사명
    imgx_rslt_ct_ctn varchar NULL, -- 영상검사결과CT내용
    resch_pat_id int4 NULL,

    CONSTRAINT ct_pk PRIMARY KEY (ct_id),
    CONSTRAINT ct_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.mr (
    mr_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    imgx_mri_dt timestamp NULL, -- 영상검사MRI검사일자
    imgx_mri_cd varchar NULL, -- 영상검사MRI검사코드
    imgx_mri_nm varchar NULL, -- 영상검사MRI검사명
    imgx_rslt_mri_ctn varchar NULL, -- 영상검사결과MRI내용
    resch_pat_id int4 NULL,

    CONSTRAINT mr_pk PRIMARY KEY (mr_id),
    CONSTRAINT mr_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.petct (
    petct_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    imgx_petct_dt timestamp NULL, -- 영상검사PECT검사일자
    imgx_petct_cd varchar NULL, -- 영상검사PECT검사코드
    imgx_petct_nm varchar NULL, -- 영상검사PECT검사명
    imgx_rslt_petct_ctn varchar NULL, -- 영상검사결과PECT내용
    resch_pat_id int4 NULL,

    CONSTRAINT petct_pk PRIMARY KEY (petct_id),
    CONSTRAINT petct_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.us (
    us_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    imgx_us_dt timestamp NULL, -- 영상검사US검사일자
    imgx_us_cd varchar NULL, -- 영상검사US검사코드
    imgx_us_nm varchar NULL, -- 영상검사US검사명
    imgx_rslt_us_ctn varchar NULL, -- 영상검사결과US내용
    resch_pat_id int4 NULL,

    CONSTRAINT us_pk PRIMARY KEY (us_id),
    CONSTRAINT us_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.ercp (
    ercp_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    ercp_act_dt timestamp NULL, -- ERCP시행일자
    ercp_act_cd varchar NULL, -- ERCP시행코드
    ercp_act_nm varchar NULL, -- ERCP시행명
    ercp_act_rslt_ctn varchar NULL, -- ERCP시행결과내용
    resch_pat_id int4 NULL,

    CONSTRAINT ercp_pk PRIMARY KEY (ercp_id),
    CONSTRAINT ercp_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.eus (
    eus_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    eus_act_dt timestamp NULL, -- EUS시행일자
    eus_act_cd varchar NULL, -- EUS시행코드
    eus_act_nm varchar NULL, -- EUS시행명
    eus_act_rslt_ctn varchar NULL, -- EUS시행결과내용
    resch_pat_id int4 NULL,

    CONSTRAINT eus_pk PRIMARY KEY (eus_id),
    CONSTRAINT eus_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.ctx (
    ctx_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    phrm_cd varchar NULL, -- 약품코드
    inhosp_medi_eng_nm varchar NULL, -- 투약약품명
    medi_ingre_nm varchar NULL, -- 약품성분명
    ordr_strt_dt timestamp NULL, -- 처방시작일자
    ordr_end_dt timestamp NULL, -- 처방종료일자
    ordr_days int4 NULL, -- 처방일수
    resch_pat_id int4 NULL,

    CONSTRAINT ctx_pk PRIMARY KEY (ctx_id),
    CONSTRAINT ctx_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS public.rtx (
    rtx_id int4 NOT NULL GENERATED ALWAYS AS IDENTITY,
    rtx_nm varchar NULL, -- 방사선치료명
    daily_tret_lorr int4 NULL, -- 일별치료선량(cGy)
    tot_tret_cnt int4 NULL, -- 총 치료회수
    tot_tret_lorr int4 NULL, -- 총 치료선량(Gy)
    rad_tret_st_dt timestamp NULL, -- 방사선치료시작일자
    rad_tret_end_dt timestamp NULL, -- 방사선치료종료일자

    resch_pat_id int4 NULL,

    CONSTRAINT rtx_pk PRIMARY KEY (rtx_id),
    CONSTRAINT rtx_patient_fk FOREIGN KEY (resch_pat_id) REFERENCES public.patient(resch_pat_id) ON DELETE CASCADE
);
