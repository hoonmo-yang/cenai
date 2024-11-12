from datetime import datetime
from pydantic import BaseModel, Field


class Batch(BaseModel):
    name: str = Field(description="Batch name")
    index: int = Field(ge=0, description="Chronological order per batch kind")

    date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )

    @property
    def batch_id(self) -> str:
        return f"{self.name}_{self.date}_{self.index:03d}"


batch = Batch(
    name="hi",
    index=1,
)

print(batch.name)
print(batch.index)
print(batch.date)
print(batch.batch_id)