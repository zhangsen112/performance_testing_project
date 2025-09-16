class PerformanceIndexInfo:
    def __init__(self):
        self.http_response_time: float | None = None
        self.page_load_time: float | None = None
        self.qps: float | None = None
        # value range: [0, 1]
        self.success_rate: float | None = None

        self.sql_response_time: float | None = None

        self.redis_response_time: float | None = None

        self.dns_response_time: float | None = None

    def fix_values(self, requests_per_second: float):
        if self.http_response_time is not None and self.http_response_time < 0:
            self.http_response_time = -self.http_response_time

        if self.page_load_time is not None and self.page_load_time < 0:
            self.page_load_time = -self.page_load_time

        if self.qps is not None and self.qps > requests_per_second:
            self.qps = requests_per_second

        if self.success_rate is None:
            if self.qps is not None:
                self.success_rate = self.qps / requests_per_second
        elif self.success_rate < 0:
            self.success_rate = 0
        elif self.success_rate > 1:
            self.success_rate = 1

        if self.sql_response_time is not None and self.sql_response_time < 0:
            self.sql_response_time = -self.sql_response_time

        if self.redis_response_time is not None and self.redis_response_time < 0:
            self.redis_response_time = -self.redis_response_time

        if self.dns_response_time is not None and self.dns_response_time < 0:
            self.dns_response_time = -self.dns_response_time
