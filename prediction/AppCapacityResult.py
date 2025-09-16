from prediction.PerformanceIndexInfo import PerformanceIndexInfo
from prediction.ServerUsage import ServerUsage


class AppCapacityResult(PerformanceIndexInfo):
    def __init__(self):
        super(AppCapacityResult, self).__init__()

        self.server_usage: list[ServerUsage] | None = None

    def fix_values(self, requests_per_second: float):
        super(AppCapacityResult, self).fix_values(requests_per_second)

        if self.server_usage is not None:
            for server_usage in self.server_usage:
                server_usage.fix_values()
