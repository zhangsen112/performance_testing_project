from datetime import datetime

from pydantic import BaseModel, Field

from model.Server import ServerConfig, ServerUsage, LooseServerUsage


class ServerCapacityPredictTaskWithoutID(BaseModel):
    name: str = Field(min_length=1)
    business_id: str = Field(min_length=24, max_length=24)
    business_index: str = Field(pattern=r'http_response_time|sql_response_time'
                                        r'|redis_response_time|dns_response_time|qps')
    requests_per_second: float = Field(gt=0)
    success_rate: float | None = Field(ge=0, le=100)

    http_response_time: float | None = Field(gt=0)
    qps: float | None = Field(gt=0)
    sql_response_time: float | None = Field(gt=0)
    redis_response_time: float | None = Field(gt=0)
    dns_response_time: float | None = Field(gt=0)

    server_usage_limit: list[LooseServerUsage | None] | None = Field(min_length=3, max_length=3)

    create_time: datetime | None

    train_id: str = Field(min_length=24, max_length=24)


class ServerCapacityPredictTask(ServerCapacityPredictTaskWithoutID):
    id: str = Field(min_length=24, max_length=24)


class ServerCapacityPredictResultWithoutID(BaseModel):
    task_id: str = Field(min_length=24, max_length=24)
    task: ServerCapacityPredictTaskWithoutID

    server_config: list[ServerConfig] = Field(min_length=3, max_length=3)
    
    server_usage: list[ServerUsage] | None = Field(min_length=3, max_length=3)

    create_time: datetime | None


class ServerCapacityPredictResult(ServerCapacityPredictResultWithoutID):
    id: str = Field(min_length=24, max_length=24)
