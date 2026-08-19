import os
import sys
import yaml
import platform
import gc
import resource
import json
import time
import threading
import urllib.request
from contextlib import contextmanager

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_INITIALIZED = True
except Exception:
    _NVML_INITIALIZED = False


def discover_ollama_pids() -> list:
    """
    Scans processes ONCE to discover PIDs associated with Ollama daemon/runner.
    """
    ollama_pids = []
    if psutil is None:
        return ollama_pids
    try:
        main_pid = os.getpid()
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                pname = (p.info.get('name') or '').lower()
                pcmd = ' '.join(p.info.get('cmdline') or []).lower()
                if ('ollama' in pname or 'ollama' in pcmd) and p.info['pid'] != main_pid:
                    ollama_pids.append(p.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass
    return ollama_pids


class MemorySampler(threading.Thread):
    """
    Background polling thread (50ms interval) that samples dynamic GPU VRAM
    and Process-Tree System RAM (Python + Ollama) using cached PIDs.
    """
    def __init__(self, device_simulator, sample_interval_s: float = 0.05):
        super().__init__(daemon=True)
        self.simulator = device_simulator
        self.sample_interval_s = sample_interval_s
        self._stop_event = threading.Event()
        
        # Discover and cache Ollama PIDs ONCE to avoid continuous /proc scanning
        self.cached_ollama_pids = discover_ollama_pids()
        self.main_pid = os.getpid()
        
        self.max_vram_gb = self.simulator.get_device_vram_gb()
        
        init_python_ram, init_system_ram = self._sample_ram()
        self.max_python_ram_gb = init_python_ram
        self.max_system_ram_gb = init_system_ram

    def _sample_ram(self) -> tuple:
        """
        Samples Python RSS and System RSS using cached PIDs (O(1) lookup).
        """
        python_rss = 0
        ollama_rss = 0
        
        if psutil is not None:
            try:
                main_proc = psutil.Process(self.main_pid)
                python_rss += main_proc.memory_info().rss
                for child in main_proc.children(recursive=True):
                    try:
                        python_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
                for pid in self.cached_ollama_pids:
                    try:
                        proc = psutil.Process(pid)
                        ollama_rss += proc.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception:
                rusage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                python_rss = rusage_kb * 1024
        else:
            rusage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            python_rss = rusage_kb * 1024

        python_gb = python_rss / (1024 ** 3)
        system_gb = (python_rss + ollama_rss) / (1024 ** 3)
        return python_gb, system_gb

    def run(self):
        while not self._stop_event.is_set():
            curr_vram = self.simulator.get_device_vram_gb()
            curr_py_ram, curr_sys_ram = self._sample_ram()
            
            if curr_vram > self.max_vram_gb:
                self.max_vram_gb = curr_vram
            if curr_py_ram > self.max_python_ram_gb:
                self.max_python_ram_gb = curr_py_ram
            if curr_sys_ram > self.max_system_ram_gb:
                self.max_system_ram_gb = curr_sys_ram
                
            time.sleep(self.sample_interval_s)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=1.0)
        
    @property
    def edge_model_vram_gb(self) -> float:
        """
        Calculated Edge VRAM footprint: Model Weight VRAM (Ollama API) + Formula KV Cache VRAM (num_ctx) + PyTorch Local VRAM.
        """
        return round(self.simulator.get_total_vram_gb(), 4)



    @property
    def pipeline_python_ram_gb(self) -> float:
        """
        Peak RSS RAM consumed strictly by the Edge-RAG Python process.
        """
        return round(self.max_python_ram_gb, 4)

    @property
    def system_total_rss_gb(self) -> float:
        """
        Combined active process RSS working set (Python + Ollama).
        """
        return round(self.max_system_ram_gb, 4)


class DeviceSimulator:
    """
    Enforces hardware profiles (e.g. 8GB VRAM cap) on the host machine
    to simulate Edge device constraints during evaluation.
    """
    def __init__(self, config_path: str = "configs/hardware_profiles.yaml"):
        if config_path and yaml:
            try:
                with open(config_path, "r") as f:
                    self.config = yaml.safe_load(f)
            except Exception:
                self.config = {}
        else:
            self.config = {}

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
        Forces a hard reset of GPU cache, memory stats, and Python garbage collection.
        """
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def get_device_vram_gb(self) -> float:
        """
        Queries physical GPU device VRAM used (in GB) via NVML or PyTorch CUDA.
        """
        if _NVML_INITIALIZED:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return info.used / (1024 ** 3)
            except Exception:
                pass
        if torch and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0

    def get_llm_server_vram_gb(self) -> float:
        """
        Queries Ollama server API (http://localhost:11434/api/ps) to retrieve
        the active model's size_vram in GB. Returns 0.0 if server is un-contactable.
        """
        try:
            req = urllib.request.Request("http://localhost:11434/api/ps", headers={"User-Agent": "Edge-RAG"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = data.get("models", [])
                    total_vram_bytes = sum(m.get("size_vram", 0) for m in models)
                    return total_vram_bytes / (1024 ** 3)
        except Exception:
            pass
        return 0.0

    def calculate_kv_cache_vram_gb(self, num_ctx: int = 32768, n_layers: int = 28, n_kv_heads: int = 2, head_dim: int = 128, bytes_per_elem: int = 2) -> float:
        """
        Calculates analytical KV Cache VRAM allocation for a given context window length.
        Formula: (2 * n_layers * n_kv_heads * head_dim * num_ctx * bytes_per_elem) / (1024^3)
        """
        kv_bytes = 2 * n_layers * n_kv_heads * head_dim * num_ctx * bytes_per_elem
        return kv_bytes / (1024 ** 3)

    def get_pytorch_vram_gb(self) -> float:
        """
        Returns local PyTorch CUDA allocated memory in GB.
        """
        if torch and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        return 0.0

    def get_total_vram_gb(self, model_name: str = "qwen3.5-2b") -> float:
        """
        Calculates analytical total Edge VRAM:
        Ollama Model Weight VRAM (from API) + Analytical KV Cache VRAM (from num_ctx formula) + PyTorch Local VRAM.
        """
        ollama_weight_vram = self.get_llm_server_vram_gb()
        
        # Read num_ctx from configs/models.yaml if available
        num_ctx = 32768
        try:
            if os.path.exists("configs/models.yaml"):
                with open("configs/models.yaml", "r") as f:
                    models_cfg = yaml.safe_load(f).get("models", {})
                    num_ctx = models_cfg.get(model_name, {}).get("context_window", 32768)
        except Exception:
            pass
            
        kv_cache_vram = self.calculate_kv_cache_vram_gb(num_ctx=num_ctx)
        pytorch_vram = self.get_pytorch_vram_gb()
        
        return ollama_weight_vram + kv_cache_vram + pytorch_vram


    @contextmanager
    def sample_query_memory(self, sample_interval_s: float = 0.05):
        """
        Context manager that launches an active MemorySampler background thread
        during query execution and returns 3-tier memory metrics.
        """
        sampler = MemorySampler(self, sample_interval_s=sample_interval_s)
        sampler.start()
        try:
            yield sampler
        finally:
            sampler.stop()

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
