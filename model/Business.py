from pydantic import BaseModel, Field


class BusinessWithoutID(BaseModel):
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    sql: str | None

    dataset_id: str = Field(min_length=24, max_length=24)


class Business(BusinessWithoutID):
    id: str = Field(min_length=24, max_length=24)
