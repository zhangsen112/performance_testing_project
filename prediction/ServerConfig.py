# COMMON_CPU_CORES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 1024, 2048]
# COMMON_MEM_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# COMMON_DISK_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

MIN_CPU_CORES = 2
MIN_MEM_SIZES = 4
MIN_DISK_SIZES = 4


class ServerConfig:
    def __init__(self):
        self.cpu: int = 0
        self.memory: int = 0
        self.disk: int = 0

    def fix_values(self):
        if self.cpu is None or self.cpu <= 0:
            self.cpu = MIN_CPU_CORES
        if self.memory is None or self.memory <= 0:
            self.memory = MIN_MEM_SIZES
        if self.disk is None or self.disk <= 0:
            self.disk = MIN_DISK_SIZES
