import os
import torch
from data.custom import Earth, Zilionis
from matplotlib import pyplot as plt

from data.handle_data import data_forward
from util import get_sc_kwargs, cmap_labels, transform_axes


def latent_space(model,
                 dataloader,
                 cmap="tab10",
                 dataset=None,
                 output_path=None,
                 writer=None):
    print("[Analyse] latent representation...")

    _, outputs, latent_activations, labels = data_forward(model, dataloader)

    """
    Plotting
    """

    latent_activations = latent_activations.detach().cpu()
    outputs = outputs.detach().cpu()
    labels = labels.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5,5))
    # fig.suptitle("Latent Space")

    ax.set_aspect("equal")
    ax.axis("off")

    if model.latent_dim > 1:
        scatter = ax.scatter(latent_activations[:, 0],
                             latent_activations[:, 1],
                             **get_sc_kwargs(),
                             c=labels,
                             cmap=cmap)
    else:
        scatter = ax.scatter(outputs[:, 0],
                             outputs[:, 1],
                             **get_sc_kwargs(),
                             c=labels,
                             cmap="tab10")

    if dataset == "Earth":
        handles, _ = scatter.legend_elements()

        string_labels = Earth().transform_labels(labels)
        ax.legend(handles, string_labels, title="labels", loc="center left", bbox_to_anchor=(1, 0.5))
    elif dataset == "Zilionis":
        handles, _ = scatter.legend_elements()

        string_labels = Zilionis().transform_labels("/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/zilionis")
        ax.legend(handles, string_labels, title="labels", loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(*scatter.legend_elements(), title="labels", loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_aspect("equal")

    if model.latent_dim == 1:
        fig_grass = plt.figure()
        plt.scatter(latent_activations[::10], torch.zeros_like(latent_activations[::10]), c=labels[::10], marker=".")

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    if writer is not None:
        writer.add_figure("latent space", fig)
        if model.latent_dim == 1:
            writer.add_figure("latent space/grass", fig_grass)


def plot_dataset(model,
                 dataloader,
                 input_dim=None,
                 output_path=None,
                 writer=None):
    print("[Analyse] Dataset")

    if input_dim != 3:
        return

    inputs, _, _, labels = data_forward(model, dataloader)

    """
    PLOTTING
    """

    inputs = inputs.detach().cpu()
    labels = labels.detach().cpu()

    sc_kwargs = get_sc_kwargs()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c=labels, cmap="tab10", s=10, marker=".", alpha=.4)
    ax.view_init(azim=20)
    transform_axes(ax)

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()


def reconstruction(model,
                   dataloader,
                   input_dim=None,
                   output_path=None,
                   writer=None):
    print("[Analyse] Reconstruction")

    # The MNIST case
    if input_dim == 784:
        inputs, outputs, latent_activations, labels = data_forward(model, dataloader)
        inputs = inputs[0].view(28, 28)
        outputs = outputs[0].view(28, 28)
    else:
        inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    """
    PLOTTING
    """

    inputs = inputs.detach().cpu()
    outputs = outputs.detach().cpu()
    latent_activations = latent_activations.detach().cpu()

    sc_kwargs = get_sc_kwargs()

    if input_dim == 784:
        fig_sum, (ax1, ax3) = plt.subplots(1, 2, figsize=(5, 5))
        # fig_sum.suptitle(f"Input/Output comparison")

        ax1.imshow(inputs)
        ax1.set_aspect("equal")
        # ax1.set_title("input")

        ax3.imshow(outputs)
        ax3.set_aspect("equal")
        # ax3.set_title(r"output")
    elif input_dim == 3:
        fig_sum = plt.figure()
        ax1 = fig_sum.add_subplot(1, 3, 1, projection='3d')
        ax1.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c=labels, cmap="viridis", s=10, marker=".", alpha=.4)
        ax1.view_init(azim=20)

        ax2 = fig_sum.add_subplot(1, 3, 2)
        ax2.scatter(latent_activations[:, 0], latent_activations[:, 1], **sc_kwargs, c=labels)
        ax2.set_yticks([], [])
        ax2.set_xticks([], [])
        ax2.set_aspect("equal")

        ax3 = fig_sum.add_subplot(1, 3, 3, projection="3d")
        ax3.scatter(outputs[:, 0], outputs[:, 1], outputs[:, 2], c=labels, cmap="viridis", s=10, marker=".", alpha=.4)
        ax3.view_init(azim=30)

        # prepare for mesh plot in tensorboard
        labels = cmap_labels(labels)

        labels = torch.unsqueeze(labels, 0)
        inputs = torch.unsqueeze(inputs, 0)
        outputs = torch.unsqueeze(outputs, 0)

        point_size_config = {
            'material': {
                'cls': 'PointsMaterial',
                'size': .02
            }
        }

        writer.add_mesh("input", vertices=inputs, colors=labels, config_dict=point_size_config)
        writer.add_mesh("output", vertices=outputs, colors=labels, config_dict=point_size_config)

    elif input_dim == 2:
        fig_sum, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5, 5))

        ax1.scatter(inputs[:, 0], inputs[:, 1], c=labels, s=1)
        ax2.scatter(latent_activations, torch.zeros_like(latent_activations), c=labels, s=1, marker="v")
        ax3.scatter(outputs[:, 0], outputs[:, 1], c=labels, s=1)

        ax1.set_aspect("equal")
        ax3.set_aspect("equal")

    else:
        print("THROW CUSTOM ERROR")
        return

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    # write to tensorboard
    if writer is not None:
        writer.add_figure("reconstruction", fig_sum)

        # clean up tensorboard writer
        writer.flush()
        writer.close()
