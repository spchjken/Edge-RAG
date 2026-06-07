import yaml
import sys
import platform
try:
    import torch
except ImportError:
    torch = None

class DeviceSimulator:
    """
    Enforces hardware profiles (e.g. 8GB VRAM cap) on the host machine
    to simulate Edge device constraints during evaluation.
    """
    def __init__(self, config_path: str = "configs/hardware_profiles.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def enforce_vram_limit(self, profile_name: str):
        """
        Caps the PyTorch CUDA allocator to a specific fraction of total VRAM.
        """
        if not torch or not torch.cuda.is_available():
            print("[DeviceSimulator] Warning: CUDA not available. VRAM caps ignored.")
            return

        profiles = self.config.get("profiles", {})
        if profile_name not in profiles:
            raise ValueError(f"Profile {profile_name} not found in hardware configs.")
            
        fraction = profiles[profile_name].get("vram_fraction", 1.0)
        
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            print(f"[DeviceSimulator] Enforced {profile_name} profile: VRAM capped to {fraction*100}%")
        except Exception as e:
            print(f"[DeviceSimulator] Failed to set VRAM fraction: {e}")

    def clear_gpu_state(self):
        """
        Forces a hard reset of GPU cache and memory stats for independent benchmarking.
        """
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def get_environment_header(self) -> dict:
        """
        Extracts system metadata for the reproducible CSV header.
        """
        env = {
            "os": platform.system(),
            "python_version": sys.version.split(" ")[0],
            "cuda_available": False
        }
        if torch:
            env["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                env["cuda_available"] = True
                env["gpu_name"] = torch.cuda.get_device_name(0)
        return env
