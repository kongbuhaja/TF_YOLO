import os
import tensorflow as tf
import psutil, pynvml

class Env():
    def __init__(self, cfg):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        self.info = {
            "cpu": {},
            "gpu": {}
        }
        
        self.cpu_set(cfg.cpus)
        self.gpu_set(cfg.gpus)
        
        self.update_info()
        
        self.summary()

    def cpu_set(self, cpus):
        all_cpus = os.cpu_count()
        if isinstance(cpus, str):
            if cpus == "all":
                cpus = f"0-{all_cpus-1}"
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

        self.activate_cpus = cpu_set

        try:
            os.sched_setaffinity(0, cpu_set)
        except AttributeError:
            pass
        
        num_cores = len(cpu_set)
        if num_cores > 0:
            tf.config.threading.set_intra_op_parallelism_threads(num_cores)
            tf.config.threading.set_inter_op_parallelism_threads(num_cores)

        allowed_count = len(self.activate_cpus)
        allowed_str = self._elements_to_range(self.activate_cpus)

        self.info["cpus"] = {
            "physical_cores": all_cpus,
            "active_cores_range": allowed_str,
            "active_cores_count": allowed_count
        }
        self._update_cpu_info()

    def gpu_set(self, gpus):
        pynvml.nvmlInit()
        all_gpus = tf.config.list_physical_devices("GPU")
        try:
            if gpus == "all":
                selected_gpus = all_gpus
            elif not gpus: # None or empty
                selected_gpus = []
            elif isinstance(gpus, int):
                selected_gpus = [all_gpus[gpus]]
            else:
                selected_gpus = [all_gpus[i] for i in gpus]

            tf.config.set_visible_devices(selected_gpus, "GPU")
            for gpu in selected_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            physical_gpus = tf.config.list_physical_devices('GPU')
            visible_gpus = tf.config.get_visible_devices('GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            
            phys_to_log_map = {}
            for p_dev, l_dev in zip(visible_gpus, logical_gpus):
                phys_to_log_map[p_dev.name] = l_dev

            gpus_data = self._get_gpus_data()
            capacity = 0.0
            
            data = dict()
            for i, p_gpu in enumerate(physical_gpus):
                gpu_data = gpus_data.get(i, {})

                total_mem = gpu_data.get("total_mem", 0.0) # MiB
                current_mem = gpu_data.get("used_mem", 0.0) # MiB
                temp = gpu_data.get("temp", 0)
                model_name = gpu_data.get("name", "Unknown GPU")

                capacity += total_mem
                usage = (current_mem / total_mem * 100) if total_mem > 0 else 0.0

                is_activate = False
                tf_device_name = "(Disabled)"

                if any(p_gpu.name == v.name for v in visible_gpus):
                    is_activate = True
                    if p_gpu.name in phys_to_log_map:
                        l_dev = phys_to_log_map[p_gpu.name]
                        tf_device_name = l_dev.name
                
                data[i] = {"id":i,
                           "is_activate": is_activate,
                           "tf_name": tf_device_name,
                           "model_name": model_name,
                           "used_mem": current_mem,
                           "total_mem": total_mem,
                           "usage": usage,
                           "temp": temp}
                
            self.info["gpus"] = {"physical_count": len(physical_gpus),
                                 "logical_count": len(logical_gpus),
                                 "capacity": capacity,
                                 "data": data}

        except RuntimeError as e:
            print(f"[Env Error] GPU Setup failed: {e}")

    def update_info(self):
        self._update_cpu_info()
        self._update_gpu_info()

    def _update_cpu_info(self):
        self._update_cpu_usage()
        self._update_cpu_temp()

    def _update_cpu_usage(self, interval=None):
        try:
            per_core_usages = psutil.cpu_percent(interval=interval, percpu=True)
            max_core_idx = len(per_core_usages) - 1
            allowed_usages = [per_core_usages[i] for i in self.activate_cpus if i <= max_core_idx]

            if allowed_usages:
                self.info["cpus"]["usage"] = sum(allowed_usages)/len(allowed_usages)
            else:
                self.info["cpus"]["usage"] = 0.0
        except Exception:
            self.info["cpus"]["usage"] = 0.0

    def _update_cpu_temp(self):
        if psutil is None:
            return "N/A (Install psutil)"
        
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                self.info["cpus"]["temp"] = "N/A"
            
            max_temp = 0.0
            found = False
            for name, entries in temps.items():
                for entry in entries:
                    if hasattr(entry, 'current') and entry.current:
                        max_temp = max(max_temp, entry.current)
                        found = True
            
            self.info["cpus"]["temp"] = max_temp if found else "N/A"
        except Exception:
            self.info["cpus"]["temp"] = "N/A"

    def _update_gpu_info(self):
        physical_gpus = tf.config.list_physical_devices('GPU')

        gpus_data = self._get_gpus_data()
        used_mem = 0
        temps = []

        for i, p_gpu in enumerate(physical_gpus):
            gpu_data = gpus_data.get(i, {})
            
            total_mem = gpu_data.get("total_mem", 0.0) # MiB
            current_mem = gpu_data.get("used_mem", 0.0) # MiB
            temp = gpu_data.get("temp", 0)
            
            usage = (current_mem / total_mem * 100) if total_mem > 0 else 0.0
            used_mem += current_mem
            temps.append(temp)
            
            self.info["gpus"]["data"][i]["current_mem"] = current_mem
            self.info["gpus"]["data"][i]["usage"] = usage
            self.info["gpus"]["data"][i]["temp"] = temp
        
        self.info["gpus"]["used_mem"] = used_mem
        self.info["gpus"]["temp"] = sum(temps)/len(temps)

    def _get_gpus_data(self):      
        data = {}
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            
            data[i] = {"total_mem": mem_info.total / (1024**2),
                        "used_mem": mem_info.used / (1024**2),
                        "temp": temp,
                        "name": name
            }
            
        return data

    def summary(self):
        cpus_info = self.info["cpus"]
        gpus_info = self.info["gpus"]
        
        head = "📢 Hardware Setting Summary"
        length = 90
        
        print("="*length)
        print(" "*((length-len(head))//2) + head)
        print("="*length)
        
        # [CPU Info]
        print("[CPUs Info]")
        cpus_usage = self._get_cpu_usage()
        cpus_temp = self._get_cpu_temp()
        print(f"  - System Physical Cores : {cpus_info['physical_cores']}")
        print(f"  - Activate Cores (Affinity) : {cpus_info['active_cores_range']} (Usage: {cpus_usage}) | Temp: {cpus_temp}")
        
        # [GPU Info]
        print("\n[GPUs Info]")
        print(f"  - Physical GPUs Detected : {gpus_info['physical_count']}")
        print(f"  - Logical GPUs (Visible): {gpus_info['logical_count']}")
        
        if gpus_info["data"]:
            for gpu_info in gpus_info["data"].values():
                print(f"    Target {gpu_info['id']}: {gpu_info['model_name']} ({gpu_info['tf_name']})")
                
                if gpu_info['is_activate']:
                    status_icon = "✅"
                    gpu_usage = f"{gpu_info['used_mem']:.1f} / {gpu_info['total_mem']:.1f}MB (Usage: {gpu_info['usage']:.1f}%)"
                else:
                    status_icon = "❌"
                    gpu_usage = "Disabled (0.0MB Used)"
                gpu_temp = gpu_info["temp"]
                print(f"      - {status_icon} GPU Memory: {gpu_usage} | Temp: {gpu_temp}°C")
        else:
            print("    (Running on CPU Mode)")
            
        print("="*length + "\n")

    def _elements_to_range(self, elements):
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
    
    def get_info(self):
        return {"CPU": f"{self._get_cpu_usage(p=0)}/{self._get_cpu_temp(p=0)}",
                "GPU": f"{self._get_gpu_mem(gb=True, p=0)}/{self._get_gpu_temp(p=0)}"}
    
    def _get_cpu_usage(self, p=1):
        return f"{self.info['cpus']['usage']:.{p}f}%"

    def _get_cpu_temp(self, p=1):
        if self.info['cpus']['temp'] == "N/A":
            return "N/A"
        return f"{self.info['cpus']['temp']:.{p}f}°C"
    
    def _get_gpu_usage(self, is_percent=False, gb=True, p=1):
        div = 1024 if gb else 1
        byte = "G" if gb else "M"

        if is_percent:
            return f"{self.info['gpus']['used_mem']/self.info['gpus']['capacity']:.{p}f}%"
        else:
            return f"{self.info['gpus']['used_mem']/div:.{p}f}/{self.info['gpus']['capacity']/div:.{p}f}{byte}"
        
    def _get_gpu_mem(self, gb=True, p=1):
        div = 1024 if gb else 1
        byte = "G" if gb else "M"

        return f"{self.info['gpus']['used_mem']/div:.{p}f}{byte}"
    
    def _get_gpu_temp(self, p=1):
        return f"{self.info['gpus']['temp']:.{p}f}°C"
    
    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except TypeError:
            pass