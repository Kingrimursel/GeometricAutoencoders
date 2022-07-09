import os
import pickle
import sys
from decimal import Decimal

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

sys.path.insert(0, '//')

from tests.evaluation import evaluate

commands = sys.argv

alpha = float(commands[1])
beta = float(commands[2])
delta = float(commands[3])
gamma = float(commands[4])
epsilon = float(commands[5])
writer_dir = commands[6]
epochs = int(commands[7])
save_string = commands[8]
model_path = commands[9]

if save_string == "True":
    save = True
else:
    save = False

if model_path == "None":
    model_path = None

# save distances to file
with open("/export/home/pnazari/workspace/AutoEncoderVisualization/tests/optimization/log/data.pkl", "rb") as f:
    data = pickle.load(f)

data["running"] += 1

with open("/export/home/pnazari/workspace/AutoEncoderVisualization/tests/optimization/log/data.pkl", "wb") as f:
    pickle.dump(data, f)

print(
    f"[OPT] starting with {Decimal(alpha):.4e}, {Decimal(beta):.4e}, {Decimal(delta):.4e} {Decimal(gamma):.4e}")

evaluate(alpha=alpha, beta=beta, delta=delta, gamma=gamma, epsilon=epsilon,
         mode="normal",
         model_path=model_path,
         create_video=False,
         std=1,
         n_gaussian_samples=10,
         n_origin_samples=64,
         writer_dir=writer_dir,
         epochs=epochs,
         save=save)

data["running"] -= 1

with open("/export/home/pnazari/workspace/AutoEncoderVisualization/tests/optimization/log/data.pkl", "wb") as f:
    pickle.dump(data, f)

print("[OPT] finished")
