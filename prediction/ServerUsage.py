class ServerUsage:
    def __init__(self, cpu_usage: float = 0, memory_usage: float = 0, disk_usage: float = 0):
        self.cpu_usage: float = cpu_usage
        self.memory_usage: float = memory_usage
        self.disk_usage: float = disk_usage

    def fix_values(self):
        # value range: [0, 100]

        if self.cpu_usage is None or self.cpu_usage < 0:
            self.cpu_usage = 0
        elif self.cpu_usage > 100:
            self.cpu_usage = 100

        if self.memory_usage is None or self.memory_usage < 0:
            self.memory_usage = 0
        elif self.memory_usage > 100:
            self.memory_usage = 100

        if self.disk_usage is None or self.disk_usage < 0:
            self.disk_usage = 0
        elif self.disk_usage > 100:
            self.disk_usage = 100
