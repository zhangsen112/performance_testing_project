from datetime import datetime

from pydantic import BaseModel, Field


class DatasetWithoutID(BaseModel):
    name: str
    description: str | None
    create_time: datetime | None


class DataSetGet(DatasetWithoutID):
    id: str = Field(min_length=24, max_length=24)
    dataset_file_name: str
    dataset_original_file_name: str
    dataset_content_type: str | None
    dataset_line_count: int
    dataset_url_list: list[str]

