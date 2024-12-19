from pydantic import BaseModel, Field


class PDACResultClassify(BaseModel):
    class Config:
        title = "CT 판독문 분류 결과"
        description = "CT 판독문 분류 결과"

    type: str = Field(
        ...,
        description="AI 분류기가 예측한 CT 판독문의 유형",
    )

    reason: str = Field(
        ...,
        description="AI 분류기가 CT 판독문의 유형을 예측한 근거",
    )
