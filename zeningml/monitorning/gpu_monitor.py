from pynvml import *

class GPUMonitor:
    def __init__(self):
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()
        
    def get_utilization(self):
        metrics = {}
        for i in range(self.device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            util = nvmlDeviceGetUtilizationRates(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)
            metrics[f"gpu_{i}"] = {
                'gpu_util': util.gpu,
                'mem_util': mem.used/mem.total * 100,
                'temperature': nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            }
        return metrics

    def alert_overutilization(self, threshold=90):
        metrics = self.get_utilization()
        return any(gpu['gpu_util'] > threshold for gpu in metrics.values())