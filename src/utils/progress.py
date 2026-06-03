import time, sys

class ProgressBar:
    def __init__(self, iterable, task="", split="", headers=None, total=None, ncols=12, min_interval=0.5):
        self.iterable = iterable
        self.headers = headers if headers else ["Epoch", "GPU_Mem", "Cls_Loss", "Box_Loss", "Dfl_Loss"]
        self.task = task.upper()
        self.split = split.upper()
        self.ncols = ncols
        self.min_interval = min_interval
        self.is_ipython = False
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from IPython.display import clear_output
                self.clear_output = clear_output
                self.is_ipython = True
        except ImportError:
            self.is_ipython = False

        if self.is_ipython:
            self.print = self.ipython_print
            self.clear = self.ipython_clear
        else:
            self.print = self.terminal_print
            self.clear = self.terminal_clear
        
        self.total = len(iterable)
        self._initiate()
            
        self.values = {header: "" for header in self.headers}

    def __len__(self):
        return self.total

    def _initiate(self):
        self.start_time = time.time()
        self.current = 0
        self.first_print = True

    def __iter__(self):
        self._initiate()
                
        for item in self.iterable:
            yield item
            self.current += 1
            self._print_lines()

        if self.current == self.total:
            self._print_lines(force=True)

        self.print([""])

    def set_status(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.headers:
                self.headers.append(key)
            if isinstance(value, float):
                self.values[key] = f"{value:.5f}"
            else:
                self.values[key] = str(value)

    def _col_width(self):
        w = 0
        for h in self.headers:
            w = max(w, len(h), len(str(self.values.get(h, ""))))
        return max(w + 2, 10)

    def _format_time(self, seconds):
        if seconds is None: return "??"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _print_lines(self, force=False):
        now = time.time()
        
        if not force and (now - getattr(self, "last_print_time", 0) < self.min_interval):
            return

        self.last_print_time = now 
        
        elapsed = now - self.start_time
        rate_disp = self.current / elapsed if elapsed > 0 else 0
        if rate_disp > 0:
            remaining = (self.total - self.current) / rate_disp
        else:
            remaining = 0

        percent = int((self.current / self.total) * 100) if self.total > 0 else 0
        prefix = f"{int(percent)}%|"
        suffix = f"| {self.current}/{self.total} [{self._format_time(elapsed)}<{self._format_time(remaining)}, {rate_disp:.2f}it/s]"
        
        ncols = self._col_width()
        
        width = len(self.headers) * ncols if self.headers else len(prefix) + len(suffix) + 5
        bar_len = width - len(prefix) - len(suffix)
        if bar_len < 0: bar_len = 0
        
        fill = (bar_len * self.current // self.total) if self.total > 0 else 0
        bar_char = "█" * fill + "░" * (bar_len - fill)
        
        line1 = f"{self.task:>{ncols-3}} | "
        for h in self.headers:
            line1 += f"{h:<{ncols}}"
            
        line2 = f"{self.split:>{ncols-3}} | "
        for h in self.headers:
            val = self.values.get(h, "")
            line2 += f"{val:<{ncols}}"
            
        if self.current >= self.total:
            line3 = f"{'Done':>{ncols-3}} | "
        else:
            line3 = f"{'Running':>{ncols-3}} | "
        line3 += f"{prefix}{bar_char}{suffix}"
        
        lines = [line for line in [line1, line2, line3] if line.strip()]

        self.clear(lines)
        self.print(lines)
    
    def ipython_clear(self, lines):
        self.clear_output(wait=True)

    def ipython_print(self, lines):
        print("\n".join(lines))

    def terminal_clear(self, lines):
        if not self.first_print:
            sys.stdout.write(f"\033[{len(lines)}A") 
        else:
            self.first_print = False

    def terminal_print(self, lines):
        for line in lines:
            sys.stdout.write(f"{line}\033[K\n")
