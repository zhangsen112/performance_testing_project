from datetime import datetime

from pydantic import BaseModel, Field

from model.Server import ServerConfig, ServerUsage


class AppCapacityPredictTaskWithoutID(BaseModel):
    name: str = Field(min_length=1)
    business_id: str = Field(min_length=24, max_length=24)
    business_index: str = Field(pattern=r'http_response_time|qps|success_rate|business_traffic'
                                        r'|sql_response_time|redis_response_time|dns_response_time')
    requests_per_second: float = Field(gt=0)

    server_config: list[ServerConfig] = Field(min_length=3, max_length=3)
    server_usage_limit: list[ServerUsage] | None = Field(min_length=3, max_length=3)

    create_time: datetime | None

    train_id: str = Field(min_length=24, max_length=24)


class AppCapacityPredictTask(AppCapacityPredictTaskWithoutID):
    id: str = Field(min_length=24, max_length=24)


class AppCapacityPredictResultWithoutID(BaseModel):
    task_id: str = Field(min_length=24, max_length=24)
    task: AppCapacityPredictTaskWithoutID

    http_response_time: float | None = Field(ge=0)
    qps: float | None = Field(ge=0)
    page_load_time: float | None = Field(ge=0)
    sql_response_time: float | None = Field(ge=0)
    redis_response_time: float | None = Field(ge=0)
    dns_response_time: float | None = Field(ge=0)

    server_usage: list[ServerUsage] | None = Field(min_length=3, max_length=3)

    create_time: datetime | None


class AppCapacityPredictResult(AppCapacityPredictResultWithoutID):
    id: str = Field(min_length=24, max_length=24)
