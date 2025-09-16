from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    cpu: int = Field(gt=0)
    memory: int = Field(gt=0)
    disk: int = Field(gt=0)


class ServerUsage(BaseModel):
    cpu_usage: float = Field(ge=0, le=100)
    memory_usage: float = Field(ge=0, le=100)
    disk_usage: float = Field(ge=0, le=100)


class LooseServerUsage(BaseModel):
    cpu_usage: float | None = Field(ge=0, le=100)
    memory_usage: float | None = Field(ge=0, le=100)
    disk_usage: float | None = Field(ge=0, le=100)
