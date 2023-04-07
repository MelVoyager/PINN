import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time

import dill

dill.settings['recurse'] = True


class HookedStdout:

    def __init__(self, filename) -> None:
        self.stdout = sys.stdout
        self.file = open(filename, 'w')

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def train_process(data, save_path, device, seed):
    hooked = HookedStdout(f"{save_path}/log.txt")
    sys.stdout = hooked

    import torch
    import deepxde as dde
    torch.cuda.set_device(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dde.config.set_default_float('float32')
    dde.config.set_random_seed(seed)

    get_model, train_args = dill.loads(data)
    model = get_model()
    model.train(**train_args, model_save_path=save_path)


class Trainer:

    def __init__(self, exp_name, device) -> None:
        self.exp_name = exp_name
        self.device = device.split(",")
        self.repeat = 1
        self.tasks = []

    def set_repeat(self, repeat):
        self.repeat = repeat

    def add_task(self, get_model, train_args):
        data = dill.dumps((get_model, train_args))
        self.tasks.append((data, train_args))

    def setup(self, filename, seed):
        os.makedirs(f"runs/{self.exp_name}")
        shutil.copy(filename, f"runs/{self.exp_name}/script.py.bak")
        json.dump({"seed": seed, "task": self.tasks}, open(f"runs/{self.exp_name}/config.json", 'w'), indent=4, default=lambda _: "...")

        mp.set_start_method('spawn')

    def train_all(self):
        if len(self.device) > 1:
            return self.train_all_parallel()
        # TODO: disable multi-processing when only one device is available
        device = "cuda:" + self.device[0]
        for j in range(self.repeat):
            for i, (data, _) in enumerate(self.tasks):
                seed = random.randint(0, 10**9)
                save_path = f"runs/{self.exp_name}/{i}-{j}"
                os.makedirs(save_path)

                print(f"***** Begin #{i}-{j} *****")
                p = mp.Process(target=train_process, args=(data, save_path, device, seed), daemon=True)
                p.start()
                p.join()
                print(f"*****  End #{i}-{j}  *****")

    def train_all_parallel(self):
        # maintain a pool of processes
        # do not start all processes at the same time
        # keep the number of processes equal to the number of devices
        # if a process is done, start a new one on the same device

        processes = [None] * len(self.device)
        for j in range(self.repeat):
            for i, (data, _) in enumerate(self.tasks):
                # find a free device
                for k, p in enumerate(processes):
                    if p is None:
                        device = "cuda:" + self.device[k]
                        seed = random.randint(0, 10**9)
                        save_path = f"runs/{self.exp_name}/{i}-{j}"
                        os.makedirs(save_path)

                        print(f"***** Start #{i}-{j} *****")
                        p = mp.Process(target=train_process, args=(data, save_path, device, seed), daemon=True)
                        p.start()
                        processes[k] = p
                        break
                else:
                    raise RuntimeError("No free device")

                # wait for a process to finish
                while True:
                    for k, p in enumerate(processes):
                        if p is None or not p.is_alive():
                            # free device
                            processes[k] = None
                            break
                    else:
                        time.sleep(5)
                        continue
                    break

        for p in processes:
            if p is not None:
                p.join()
