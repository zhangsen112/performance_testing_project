from datetime import datetime

from pydantic import BaseModel, Field


class TrainWithoutID(BaseModel):
    name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=24, max_length=24)
    model_type: str = Field(pattern=r'app_capacity|server_capacity')
    description: str | None
    create_time: datetime | None


class Train(TrainWithoutID):
    id: str = Field(min_length=24, max_length=24)
    state: str = Field(pattern=r'none|pending|running|success|failed')
    accuracy: float | None = Field(ge=0, le=1)
    dataset_name: str = Field(min_length=1)
