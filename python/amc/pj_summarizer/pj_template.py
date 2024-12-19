from pydantic import BaseModel, Field


class PJSummaryTemplate(BaseModel):
    class Config:
        title = "환자여정 요약문"
        description = "환자여정 요약문입니다"

    resch_pat_id: int = Field(
        ...,
        description="연구환자ID",
    )

    birth_ym: str = Field(
        ...,
        description="생년월",
    )

    sex_cd: str = Field(
        ...,
        description="성별코드",
    )

    frst_vist_dt: str = Field(
        ...,
        description="최초내원일자",
    )

    dx_dt: str = Field(
        ...,
        description="진단일자",
    )

    prmr_orgn_cd: str = Field(
        ...,
        description="원발장기코드",
    )

    mrph_diag_cd: str = Field(
        ...,
        description="형태학적진단코드",
    )

    cancer_reg_dt: str = Field(
        ...,
        description="암등록일",
    )

    type1: str = Field(
        ...,
        description="""유형 1. Initial diagnosis & staging 관점의 환자 진료 여정 요약 내용""",
    )

    type2: str = Field(
        ...,
        description="""유형 2. Follow-up for pancreatic cancer without curative resection 관점의
                    환자 진료 여정 요약 내용""",
    )

    type3: str = Field(
        ...,
        description="""유형 3. Follow-up after curative resection of pancreatic cancer 관점의
                    환자 진료 여정 요약 내용""",
    )

    type4: str = Field(
        ...,
        description="""유형 4. Follow-up for tumor recurrence after curative resection 관점의
                    환자 진료 여정 요약 내용""",
    )

    total: str = Field(
        ...,
        description="""전체적인 환자 진료 여정 요약 내용""",
    )
