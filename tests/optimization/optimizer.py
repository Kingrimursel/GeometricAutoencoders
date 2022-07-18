import os
import pickle
import sys

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import time
from subprocess import Popen
import torch

with open("/export/home/pnazari/workspace/AutoEncoderVisualization/tests/optimization/log/data.pkl", "wb") as f:
    pickle.dump({"running": 0}, f)

alphas = torch.logspace(-4, 4, 9)
betas = torch.tensor([0.])  # torch.logspace(-5, 5, 11)  # torch.hstack((torch.tensor([0]), torch.logspace(-5, 5, 11)))
deltas = torch.tensor([0.])  # torch.hstack((torch.tensor([0]), torch.logspace(-8, 2, 11)))
epsilons = torch.tensor([0.])
gammas = torch.tensor([0.])  # torch.hstack((torch.tensor([0]), torch.linspace(1e-3, 1e-2, 100)))  # torch.hstack((torch.tensor([0]), torch.logspace(-8, 1, 10)))
writer_dir = "Zilionis/alpha"

max_parallel = 4
epochs = 50
save = True
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/MNIST/ELUAutoEncoder/2022.06.17/MNIST/beta/0.0000e+4_1.0000e-1_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/FashionMNIST/ELUAutoEncoder/2022.06.22/FashionMNIST/beta/0.0000e+4_1.0000e-1_0.0000e+4_0.0000e+4.pth"
model_path = "None"

counter = 0
for alpha in alphas:
    for beta in betas:
        for delta in deltas:
            for gamma in gammas:
                for epsilon in epsilons:
                    # loop that checks every x seconds if little enough instances are running
                    while True:
                        with open(
                                "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/optimization/log/data.pkl",
                                "rb") as f:
                            data = pickle.load(f)
                            print(f"[OPT] running: {data['running']}")

                        if counter != 0:
                            time.sleep(30)
                        if data["running"] < max_parallel:
                            break

                    print(f"[OPT]: {counter + 1} of {len(alphas) * len(betas) * len(deltas)}")
                    cmd = f"python3 /export/home/pnazari/workspace/AutoEncoderVisualization/tests/optimization/handler.py " \
                          f"{alpha.item()} {beta.item()} {delta.item()} {gamma.item()} {epsilon.item()} {writer_dir} {epochs} {save} {model_path}"

                    proc = Popen([cmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

                    counter += 1
