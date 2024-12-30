from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class PJSummaryTemplate(BaseModel):
    class Config:
        title = "환자여정 요약문"
        description = "환자여정 요약문입니다"

    nickname: str = Field(
        ...,
        description="환자 가명",
    )

    ct_date: str = Field(
        ...,
        description="요약 대상인 환자가 받은 CT진단 일자",
    )

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

    type: TypeEnum = Field(
        ...,
        description="""예측 유형""",
    )

    reason: str = Field(
        ...,
        description="""유형을 에측한 이유 및 근거""",
    )

    summary: str = Field(
        ...,
        description="""유형에 기반한 환자 진료 여정 요약 내용""",
    )


class TypeEnum(str, Enum):
    NOT_AVAILABLE = "Not available"
    INSUFFICIENT_EVIDENCE = "Insufficient evidence"
    TYPE1 = "1. Initial diagnosis & staging"
    TYPE2 = "2. Follow-up for pancreatic cancer without curative resection"
    TYPE3 = "3. Follow-up after curative resection of pancreatic cancer"
    TYPE4 = "4. Follow-up for tumor recurrence after curative resection"
