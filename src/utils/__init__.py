import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from src.utils.draw import Painter
from src.utils.metric import Metrics

__all__ = (
    "Painter",
    "Metrics",
    "Env",
)

class Env():
    def __init__(self, cfg):
        self.cpu_set(cfg["cpus"])
        self.gpu_set(cfg["gpus"])
        self.summary()

    def gpu_set(self, gpus):
        all_gpus = tf.config.list_physical_devices("GPU")
        try:
            if gpus == "all":
                selected_gpus = all_gpus
            elif not gpus:
                selected_gpus = []
            elif isinstance(gpus, int):
                selected_gpus = [all_gpus[gpus]]
            else:
                selected_gpus = [all_gpus[i] for i in gpus]

            tf.config.set_visible_devices(selected_gpus, "GPU")
            for gpu in selected_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)

    def cpu_set(self, cpus):
        if isinstance(cpus, str):
            if cpus == "all":
                cpus = f"0-{os.cpu_count()-1}"
            cpus = [cpus]
        
        cpu_set = set()

        for part in cpus:
            if isinstance(part, int):
                cpu_set.add(part)
            elif isinstance(part, str) and "-" in part:
                start, end = map(int, part.split("-"))
                cpu_set.update(range(start, end+1))
            elif isinstance(part, str):
                cpu_set.add(int(part))

        try:
            os.sched_setaffinity(0, cpu_set)
        except AttributeError as e:
            print(e)
        
        num_cores = len(cpu_set)
        tf.config.threading.set_intra_op_parallelism_threads(num_cores)
        tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    
    def summary(self):
        head = "📢 Hardware Setting Summary"
        length = 80
        print("="*length)
        print(" "*((length-len(head))//2) + head)
        print("="*length)
        self.cpu_summary()
        self.gpu_summary()
        print("="*length + "\n")

    def cpu_summary(self):
        print(f"[CPU Info]")
        total_physical_cores = os.cpu_count()
        print(f"  - System Physical Cores : {total_physical_cores}")

        try:
            affinity = os.sched_getaffinity(0)
            allowed_count = len(affinity)
            allowed_str = self.elements_to_range(affinity)
            print(f"  - ✅ Active Cores (Affinity) : {allowed_str} (Total: {allowed_count})")
        except AttributeError:
            print("  - Active Cores : (os.sched_getaffinity not supported on this OS)")

    def gpu_summary(self):
        print(f"\n[GPU Info]")
        physical_gpus = tf.config.list_physical_devices('GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')

        print(f"  - Physical GPUs Detected : {len(physical_gpus)}")
        print(f"  - ✅ Logical GPUs (Visible): {len(logical_gpus)}")

        def get_gpu_total_memory_map():
            import subprocess
            try:
                # nvidia-smi로 전체 메모리 조회 (단위: MiB)
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                    encoding="utf-8"
                )
                memory_map = {}
                for line in result.strip().split('\n'):
                    idx, mem = line.split(',')
                    memory_map[int(idx)] = float(mem.strip())
                return memory_map
            except (FileNotFoundError, subprocess.CalledProcessError):
                return {}
            
        total_mem_map = get_gpu_total_memory_map()

        if logical_gpus:
            for i, gpu in enumerate(logical_gpus):
                try:
                    mem_info = tf.config.experimental.get_memory_info(gpu.name)
                    current_mem = mem_info['current'] / (1024**2)
                    total_mem = total_mem_map.get(i, 0.0)
                    ratio = (current_mem / total_mem) * 100
                    print(f"    Target {i}: {gpu.name} | Mem Usage: {current_mem:.1f}MB / {total_mem:.1f}MB ({ratio:.1f}%))")
                except:
                    print(f"    Target {i}: {gpu.name}")
        else:
            print("    (Running on CPU Mode)")

    def elements_to_range(self, elements):
        if not elements: return "None"
        cpus = sorted(list(elements))
        ranges = []
        start = cpus[0]
        prev = cpus[0]
        
        for i in range(1, len(cpus)):
            if cpus[i] != prev + 1:
                if start == prev:
                    ranges.append(f"{start}")
                else:
                    ranges.append(f"{start}-{prev}")
                start = cpus[i]
            prev = cpus[i]
        
        if start == prev:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{prev}")
            
        return ", ".join(ranges)