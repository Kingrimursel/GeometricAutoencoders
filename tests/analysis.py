import os
import torch

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

from evaluation import evaluate, knn_evaluation

# knn_evaluation("MNIST",
#               ["models/MNIST/ELUAutoEncoder/2022.06.17/MNIST/alpha",
#                "models/MNIST/ELUAutoEncoder/2022.06.17/MNIST/beta",
#                "models/MNIST/ELUAutoEncoder/2022.06.22/MNIST/gamma"],
#               writer_dir="test",
#               indices=["alpha", "beta", "gamma"],
#               labels=[r"$\mathbf{det}$", r"$\mathbf{dist}$", r"$\mathbf{indic}$"],
#               k=5)


# SADDLE
model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Saddle/DeepThinAutoEncoder/2022.06.22/Saddle/0.0000e+4_0.0000e+4_0.0000e+4_0.0000e+4.pth"

# MNIST
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/MNIST/ELUAutoEncoder/2022.06.17/MNIST/alpha/0.0000e+4_0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/MNIST/ELUAutoEncoder/2022.06.17/MNIST/alpha/1.0000e-2_0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/MNIST/ELUAutoEncoder/2022.06.17/MNIST/beta/0.0000e+4_1.0000e-1_0.0000e+4_0.0000e+4.pth"

# SPHERES
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Spheres/DeepThinAutoEncoder/2022.06.17/Spheres/alpha/0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Spheres/DeepThinAutoEncoder/2022.06.17/Spheres/alpha/1.0000e-2_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Spheres/DeepThinAutoEncoder/2022.06.17/Spheres/beta/0.0000e+4_1.0000e+2_0.0000e+4.pth"

# Earth
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Earth/DeepThinAutoEncoder/2022.06.25/Earth/alpha/0.0000e+4_0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Earth/DeepThinAutoEncoder/2022.06.25/Earth/beta/0.0000e+4_1.0000e-3_0.0000e+4_0.0000e+4.pth"

# FigureEight
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/FigureEight/DeepThinAutoEncoder/2022.06.25/FigureEight/0.0000e+4_0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/hocme/pnazari/workspace/AutoEncoderVisualization/tests/output/models/FigureEight/DeepThinAutoEncoder/2022.06.27/FigureEight/0.0000e+4_0.0000e+4_0.0000e+4_0.0000e+4.pth"

# Mammoth
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Mammoth/DeepThinAutoEncoder/2022.07.06/Mammoth/beta/0.0000e+4_0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Mammoth/DeepThinAutoEncoder/2022.07.06/Mammoth/alpha/1.0000e+1_0.0000e+4_0.0000e+4_0.0000e+4.pth"
# model_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/tests/output/models/Mammoth/DeepThinAutoEncoder/2022.07.06/Mammoth/beta/0.0000e+4_1.0000e+4_0.0000e+4_0.0000e+4.pth"

evaluate(alpha=0, beta=0., delta=0., epsilon=0., gamma=0., writer_dir="video",
         epochs=30,
         mode="vanilla",
         create_video=True,
         train=True,
         model_path=None,
         save=False,
         std=1,
         n_gaussian_samples=10,
         n_origin_samples=64
         )
