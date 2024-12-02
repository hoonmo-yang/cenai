from __future__ import annotations
from pydantic import BaseModel, Field


class PaperSummaryTemplate(BaseModel):
    class Config:
        title = "연구결과 요약문"
        description = "연구결과 요약문입니다"

    abstract: str = Field(
        ...,
        description="연구개요",
    )

    outcome: str = Field(
        ...,
        description="연구 목표대비 연구결과",
    )

    expectation: str = Field(
        ...,
        description="연구개발성과의 활용 계획 및 기대효과(연구개발결과의 중요성)",
    )

    keyword_kr: list[str] = Field(
        ...,
        description="총 5개의 중심어 (국문)",
    )

    keyword_en: list[str] = Field(
        ...,
        description="총 5개의 중심어 (영문)",
    )


class PaperResult(BaseModel):
    class Config:
        title = "Result basic class"
        description = "Result basic class"

    error: bool = False


class PaperResultSummary(PaperResult):
    class Config:
        title = "내용 요약 결과"
        description = "내용 요약 결과 (abstract, outcome, expectation)"

    summary: str = Field(
        ...,
        description="요약 결과",
    )


class PaperResultKeyword(PaperResult):
    class Config:
        title = "중심어 추출 결과"
        description = "중심어 추출 결과 (keyword)"

    keyword_kr: list[str] = Field(
        ...,
        description="국문 중심어 추출 결과",
    )

    keyword_en: list[str] = Field(
        ...,
        description="영문 중심어 추출 결과",
    )


class PaperResultSimilarity(PaperResult):
    class Config:
        title = "논문 요약 유사도 비교 점수"
        description = "논문 요약 유사도 비교 점수"

    score: float = Field(
        ...,
        description="요약 유사도 점수" 
    )

    difference: str = Field(
        ...,
        description="요약이 다른 부분에 대한 설명"
    ),


class PaperResultFail(PaperResult):
    class Config:
        title = "결과 추출 실패"
        description = "결과 추출 실패"

    message: str = Field(
        ...,
        description="결과 추출 실패 사유"
    )

    error: bool = True
