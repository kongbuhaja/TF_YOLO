from src.data.dataset import Dataset
import numpy as np
from src.data.process import Process, Augmentation
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from copy import deepcopy

class Dataloader():
    def __init__(self, cfg, data={"data":[], "dtype":""}):
        for name, value in cfg.__dict__.items():
            setattr(self, name, value)
        self.epoch = 0
        self.close_mosaic = self.epochs - self.close_mosaic + 1
        self.insert_data(data)

    def insert_data(self, data):
        self.dtype = data["dtype"]
        self.data = data["data"]
        self.indices = np.arange(len(self.data))

        # process
        preprocess = Process({"read_image": True})

        if self.dtype == "train" or self.dtype == "val":
            process = Process(self.augmentation) + Process(self.process)
        elif self.dtype == "val":
            process = Process(self.process)
        else:
            self.batch_size = 1
            process = Process({"resize_padding_with_info": self.process["resize_padding"][:2],
                                    "unsqueeze_coords": self.process["unsqueeze_coords"],
                                    "segment_to_task": self.process["segment_to_task"]})
            
        self.process = preprocess + process
        self.postprocess = Process(self.postprocess)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __iter__(self):
        self.on_epoch_start()
        return self
    
    def __next__(self):
        if self.idx >= len(self):
            self.on_epoch_end()
            raise StopIteration
        
        data = self.queue.get()

        if isinstance(data, Exception):
            print(f"Error detected from worker process: {data}")
            self.on_epoch_end()
            raise data
        
        self.idx += 1
        return data

    def _prefetch(self):
        while not self._stop_signal and self.prefetch_idx < len(self):
            try:
                batch_indices = self.indices[self.prefetch_idx * self.batch_size : (self.prefetch_idx + 1) * self.batch_size]
                
                if getattr(self.process, "use_mosaic", False):
                    sample_indices = [[idx] + np.random.choice(self.indices, 3).tolist() for idx in batch_indices]
                    batch_data = [[self.data[idx].copy() for idx in indices] for indices in sample_indices]
                else:
                    batch_data = [self.data[idx].copy() for idx in batch_indices]
                
                data = list(self.executor.map(self.process, batch_data))
                data = self.postprocess(data)

                self.queue.put(data)
                self.prefetch_idx += 1
            except Exception as e:
                self.queue.put(e)
        
    def on_epoch_start(self):
        self.idx = 0
        self.prefetch_idx = 0
        self.epoch += 1
        
        if self.dtype == "train":
            if self.shuffle:
                np.random.shuffle(self.indices)
            if self.epoch == self.close_mosaic:
                self.process.close_mosaic()

        self._stop_signal = False

        if getattr(self.process, "use_mosaic", False):
            self.executor = ProcessPoolExecutor(max_workers=self.workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.workers)

        self.queue = Queue(maxsize=self.prefetch)

        self._prefetch_thread = Thread(target=self._prefetch)
        self._prefetch_thread.daemon = True
        self._prefetch_thread.start()
    
    def on_epoch_end(self):
        if self._stop_signal:
            return 
        
        self._stop_signal = True

        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except self.queue.Empty:
                continue

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3)
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
