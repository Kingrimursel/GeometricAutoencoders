import os
from datetime import datetime
from pathlib import Path
from decimal import Decimal

import torch

from AutoEncoderVisualization.diagnostics.local_properties import decoder_knn_variance_latent, \
    encoder_knn_variance_latent, encoder_gaussian_variance
from AutoEncoderVisualization.diagnostics.map_properties import distances_from_points, circular_variance
from AutoEncoderVisualization.diagnostics.metric_properties import plot_determinants, sectional_curvature, indicatrices
from AutoEncoderVisualization.diagnostics.representations import reconstruction, latent_space, plot_dataset
from AutoEncoderVisualization.diagnostics.embedding import plot_knn_performance, classification_error_figure, \
    classification_error_table

from data.handle_data import load_data
from models import DeepThinAutoEncoder, SoftplusAE, ELUAutoEncoder, DeepThinSigmoidAutoEncoder, TestELUAutoEncoder
from conf import device, get_summary_writer, output_path
from AutoEncoderVisualization.tests.train import train_model

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

""" modify """

# currently used auto encoder
AE = DeepThinAutoEncoder
# AE = ELUAutoEncoder
# AE = TestELUAutoEncoder

# evaluate
eval_model = True

dataset = "Spheres"

""" end modify """

# set input dimensions
if dataset in ["SwissRoll", "Mammoth", "Saddle", "HyperbolicParabloid", "Earth", "Spheres"]:
    input_dim = 3
    latent_dim = 2

    train_batch_size = 512
elif dataset in ["chiocciola", "FigureEight"]:
    input_dim = 2
    latent_dim = 1

    train_batch_size = 512
elif dataset in ["MNIST", "FashionMNIST"]:
    input_dim = 784
    latent_dim = 2

    train_batch_size = 256

elif dataset in ["Zilionis"]:
    input_dim = 306
    latent_dim = 2

    train_batch_size = 256

# path to save(d) model weights
model_init_path = os.path.join(output_path, f"models/{dataset}/{AE.__name__}/init.pth")


def evaluate(alpha=None,
             beta=None,
             delta=None,
             writer_dir=None,
             epsilon=None,
             gamma=None,
             epochs=None,
             train=True,
             mode="normal",
             save=True,
             std=.2,
             n_dist_samples=20,
             n_origin_samples=None,
             init=False,
             n_gaussian_samples=5,
             model_path=None,
             create_video=False
             ):
    """
    Determine settings
    """

    # if no model path to load from is passed and model should not be trained, then there is nothing to do
    if model_path is None and train is False:
        print("[exit] neither loading nor training new model")
        return

    if alpha == 0. and beta == 0. and delta == 0. and gamma == 0. and epsilon == 0. and mode != "vanilla":
        mode = "baseline"

    # Prepare SummaryWriter
    writer = get_summary_writer(subdir=writer_dir)
    print(f"[Writer] subdir {writer_dir}")

    """
    Import Data
    """

    train_loader, test_loader = load_data(train_batch_size=train_batch_size,
                                          test_batch_size=256,
                                          dataset=dataset)

    """
    Initialize Model
    """

    # create model
    print(f"[model] move to {device}...")
    model = AE(input_shape=input_dim, latent_dim=latent_dim).to(device)

    if model_path is not None:
        print("[model] load from path...")
        model.load(model_path)
    else:
        # if in train mode
        if os.path.isfile(model_init_path) and not init:
            print("[model] load existing init...")
            model.load(model_init_path)
        else:
            if not init:
                print("[model] saving new init...")
                torch.save(model.state_dict(), model_init_path)

    if save is True:
        # set and create path for saving model
        model_path_save = os.path.join(output_path,
                                       f"models/{dataset}/{AE.__name__}/{datetime.now().strftime('%Y.%m.%d')}/{writer_dir}")
        Path(model_path_save).mkdir(parents=True, exist_ok=True)
        model_path_save = os.path.join(model_path_save,
                                       f"{Decimal(alpha):.4e}_{Decimal(beta):.4e}_{Decimal(delta):.4e}_{Decimal(gamma):.4e}.pth")
    else:
        model_path_save = None

    # set and create path for saving images
    image_save_path = os.path.join(output_path,
                                   f"graphics/{dataset}/{AE.__name__}/{datetime.now().strftime('%Y.%m.%d')}/"
                                   f"{writer_dir}/{Decimal(alpha):.4e}_{Decimal(beta):.4e}_{Decimal(delta):.4e}_{Decimal(gamma):.4e}")
    Path(image_save_path).mkdir(parents=True, exist_ok=True)

    """
    Train model
    """

    if train:
        print("[model] train...\n")

        if epochs is None:
            epochs = 30

        weight_decay = 0  # 5e-7

        # log to tensorboard
        writer.add_text("Meta",
                        f"model={model.__class__.__name__}, "
                        f"epochs={epochs}, "
                        f"train_batch_size={train_batch_size}, "
                        f"mode={mode}, "
                        f"model_path={model_path}, "
                        f"alpha={Decimal(alpha):.4e}, "
                        f"beta={Decimal(beta):.4e}, "
                        f"delta={Decimal(delta):.4e}, "
                        f"gamma={Decimal(gamma):.4e}, "
                        f"std={std}, "
                        f"n_gaussian_samples={n_gaussian_samples}, "
                        f"n_origin_samples={n_origin_samples}, "
                        f"n_dist_samples={n_dist_samples}, "
                        f"weight_decay={weight_decay}")

        # train model
        loss = train_model(model, train_loader, model_path_save, epochs=epochs, alpha=alpha, beta=beta,
                           gamma=gamma, delta=delta, epsilon=epsilon, n_dist_samples=n_dist_samples,
                           weight_decay=weight_decay, std=std, n_gaussian_samples=n_gaussian_samples,
                           n_origin_samples=n_origin_samples, writer=writer, mode=mode, create_video=create_video)

        hparams_dict = {
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "gamma": gamma,
            "std": std,
            "n_gaussian_samples": n_gaussian_samples,
            "n_origin_samples": n_origin_samples,
            "model_path": model_path,
            "epochs": epochs,
            "batch_size": train_batch_size,
            "n_dist_samples": n_dist_samples,
            "k_neighbours": weight_decay,
            "weight_decay": weight_decay
        }

        metric_dict = {
            "loss": loss
        }

        writer.add_hparams(hparam_dict=hparams_dict, metric_dict=metric_dict)

        writer.flush()
        writer.close()

    """
    Analyse Model
    """
    print("[model] analyze ...")

    if dataset == "MNIST":
        cmap = "tab10"
    else:
        cmap = "viridis"

    if eval_model:
        # preimage_of_ball(test_loader, model)
        # plot_pd(test_loader, model, 100)

        # circular_variance(test_loader,
        #                  model,
        #                  writer=writer,
        #                  output_path=os.path.join(image_save_path, "circular.png"))

        if latent_dim > 1:
            # calculate distances from multiple randomly sampled points to all other points
            distances_from_points(model,
                                  test_loader,
                                  num_rows=3,
                                  num_cols=3,
                                  output_path=os.path.join(image_save_path, "distances.png"),
                                  writer=writer)

        if latent_dim > 1:
            # calculate indicatrices
            indicatrices(model,
                         test_loader,
                         device=device,
                         num_steps=7,
                         num_gon=100,
                         output_path=os.path.join(image_save_path, "indicatrices.png"),
                         writer=writer)

        # plot reconstruction
        reconstruction(model,
                       test_loader,
                       input_dim=input_dim,
                       writer=writer,
                       output_path=os.path.join(image_save_path, "reconstruction.png")
                       )

        # plot input dataset
        plot_dataset(model,
                     test_loader,
                     input_dim=input_dim,
                     writer=writer,
                     output_path=os.path.join(image_save_path, "dataset.png")
                     )

        # plot knn evaluation
        # plot_knn_performance(model,
        #                     test_loader,
        #                     writer=writer,
        #                     output_path=os.path.join(image_save_path, "knn_similarities.png"),
        #                    k=20)

        # plot latent space
        latent_space(model,
                     test_loader,
                     cmap="tab10",
                     dataset=dataset,
                     output_path=os.path.join(image_save_path, "latent_space.png"),
                     writer=writer)

        # decoder knn variance
        decoder_knn_variance_latent(model, test_loader,
                                    scaling="log",
                                    output_path=os.path.join(image_save_path, "var_dec_lat.png"),
                                    k=100,
                                    writer=writer,
                                    # vmin=-3.8927, vmax=0.3909
                                    )

        # encoder knn variance
        encoder_knn_variance_latent(model, test_loader,
                                    scaling="log",
                                    output_path=os.path.join(image_save_path, "var_enc_lat.png"),
                                    output_path_3d=os.path.join(image_save_path, "var_enc_lat_3d.png"),
                                    k=100,
                                    writer=writer)

        # encoder gaussian variance
        encoder_gaussian_variance(model,
                                  test_loader,
                                  scaling="log",
                                  output_path=os.path.join(image_save_path, "var_enc_gaus.png"),
                                  writer=writer)

        if latent_dim > 1:
            # calculate indicatrices
            indicatrices(model,
                         test_loader,
                         device=device,
                         num_steps=7,
                         num_gon=100,
                         output_path=os.path.join(image_save_path, "indicatrices.png"),
                         writer=writer)
            # determinants
            plot_determinants(model, test_loader, batch_size=500, device=device,
                              scaling="asinh",
                              output_path_1=os.path.join(image_save_path, "det.png"),
                              output_path_2=os.path.join(image_save_path, "det_hist.png"), writer=writer)

            # sectional curvature
            sectional_curvature(model, test_loader, quantile=0.99, device=device, batch_size=100, writer=writer,
                                scaling="asinh",
                                input_dim=input_dim, output_path_1=os.path.join(image_save_path, "curv.png"),
                                output_path_2=os.path.join(image_save_path, "curv_hist.png"))

        # circular variance
        # circular_variance(test_loader,
        #                  model,
        #                  writer=writer)


def knn_evaluation(dataset,
                   model_paths,
                   writer_dir=None,
                   labels=None,
                   indices="alpha",
                   k=20):
    # Prepare SummaryWriter
    writer = get_summary_writer(subdir=writer_dir)
    print(f"[Writer] subdir {writer_dir}")

    train_loader, test_loader = load_data(train_batch_size=256,
                                          test_batch_size=256,
                                          dataset=dataset)

    AE = ELUAutoEncoder

    input_dim = 784
    latent_dim = 2

    model = AE(input_shape=input_dim, latent_dim=latent_dim).to(device)

    model_paths = [os.path.join(output_path, relative_path) for relative_path in model_paths]

    writer.add_text("Meta", f"model_paths={model_paths}")

    xranges = []

    for i, model_path in enumerate(model_paths):
        xrange = []
        for j, file in enumerate(os.listdir(model_path)):
            # there might be a readme.txt
            try:
                alpha, beta, delta, gamma = file.split("_")
                gamma = ".".join(gamma.split(".")[:-1])
            except ValueError:
                continue

            if indices[i] == "alpha":
                xrange.append(float(alpha))
            elif indices[i] == "beta":
                xrange.append(float(beta))
            elif indices[i] == "delta":
                xrange.append(float(delta))
            elif indices[i] == "gamma":
                xrange.append(float(gamma))
            else:
                return

        xrange = torch.tensor(xrange)
        xranges.append(xrange)

    # set and create path for saving images
    image_save_path = os.path.join(output_path,
                                   f"graphics/{dataset}/{AE.__name__}/{datetime.now().strftime('%Y.%m.%d')}/"
                                   f"{writer_dir}/evaluation")
    Path(image_save_path).mkdir(parents=True, exist_ok=True)

    classification_error_table(model, test_loader, model_paths, k=k, xranges=xranges, legend_labels=labels,
                               writer=writer,
                               output_path=os.path.join(image_save_path, "knn_classification_figure.png"))

    classification_error_figure(model, test_loader, model_paths, k=k, xranges=xranges, legend_labels=labels,
                                writer=writer,
                                output_path=os.path.join(image_save_path, "knn_classification_figure.png"))
