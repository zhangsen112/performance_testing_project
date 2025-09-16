from prediction.ServerConfig import ServerConfig
from prediction.ServerUsage import ServerUsage


class ServerCapacityResult:
    def __init__(self):
        self.server_config: list[ServerConfig] | None = None
        self.server_usage: list[ServerUsage] | None = None

    def fix_values(self):
        if self.server_config is not None:
            for server_config in self.server_config:
                server_config.fix_values()

        if self.server_usage is not None:
            for server_usage in self.server_usage:
                server_usage.fix_values()
