import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from conf import UPPER_EPSILON
from data.handle_data import data_forward

from diffgeo.manifolds import RiemannianManifold
from diffgeo.metrics import PullbackMetric
from diffgeo.connections import LeviCivitaConnection

from util import (get_sc_kwargs,
                  get_hull,
                  in_hull,
                  get_coordinates,
                  symlog,
                  values_in_quantile,
                  determine_scaling_fn,
                  join_plots,
                  minmax,
                  cmap_labels)


def sectional_curvature(model,
                        dataloader,
                        quantile=1.,
                        scaling="asinh",
                        grid="dataset",
                        num_steps=20,
                        device="cpu",
                        batch_size=-1,
                        writer=None,
                        input_dim=None,
                        output_path_1=None,
                        output_path_2=None,
                        **kwargs):
    print("[Analyse] sectional curvature")

    # TODO: bei der Darstellung vielleicht lokal die curvature mitteln?

    # forward pass
    inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    # initialize diffgeo objects
    pbm = PullbackMetric(2, model.decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    coordinates = get_coordinates(latent_activations, grid=grid, num_steps=num_steps).to(device)

    if batch_size == -1:
        batch_size = coordinates.shape[0]

    # calculate curvatures
    curvature = None
    for coord in torch.utils.data.DataLoader(coordinates, batch_size=batch_size):
        curv = rm.sectional_curvature(torch.tensor([1., 0.], device=device),
                                      torch.tensor([0., 1.], device=device),
                                      coord).detach()
        if curvature is None:
            curvature = curv
        else:
            curvature = torch.hstack((curvature, curv))

    scaling_fn, prefix = determine_scaling_fn(scaling)

    # scale curvature values
    scaled_curvature = scaling_fn(curvature)

    # determine curvature values that lie in quantile
    middle_idx = values_in_quantile(scaled_curvature, quantile)

    """
    PLOTTING
    """

    coordinates = coordinates.detach().cpu()
    labels = labels.detach()
    latent_activations = latent_activations.detach().cpu()
    scaled_curvature = scaled_curvature.detach().cpu()
    outputs = outputs.detach().cpu()
    curvature = curvature.detach().cpu()

    # color-coded
    fig_col, ax_col = plt.subplots(figsize=(5, 5))

    # fig_col.suptitle(f"{prefix}Sectional Curvature")
    ax_col.axis("off")

    scatter_col = ax_col.scatter(coordinates[:, 0][middle_idx],
                                 coordinates[:, 1][middle_idx],
                                 c=scaled_curvature[middle_idx],
                                 cmap="viridis",
                                 # s=10,
                                 # marker=".",
                                 # alpha=0.5,
                                 **get_sc_kwargs(),
                                 **kwargs)

    divider = make_axes_locatable(ax_col)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    sm = ScalarMappable()
    sm.set_array(curvature)

    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_alpha(0.5)
    cbar.draw_all()

    ax_col.set_aspect("equal")

    if input_dim == 3:
        fig_col_3d = plt.figure()
        ax_col_3d = fig_col_3d.add_subplot(1, 1, 1, projection="3d")
        scatter_col_3d = ax_col_3d.scatter(outputs[:, 0][middle_idx],
                                           outputs[:, 1][middle_idx],
                                           outputs[:, 2][middle_idx],
                                           c=scaled_curvature[middle_idx],
                                           s=10,
                                           marker=".",
                                           alpha=.4,
                                           **kwargs)
        fig_col_3d.colorbar(scatter_col_3d, ax=ax_col_3d)

        # Hide grid lines
        ax_col_3d.grid(False)

        # Hide axes ticks
        ax_col_3d.set_axis_off()

        ax_col_3d.view_init(azim=-60, elev=10)

    else:
        fig_col_3d = None

    if output_path_1 is not None:
        plt.savefig(output_path_1, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    # histogram
    fig_hist, ax_hist = plt.subplots(figsize=(5, 5))
    # fig_hist.suptitle(f"{prefix}Curvature Distribution")

    ax_hist.hist(scaled_curvature[middle_idx].numpy(), bins=40, density=True, alpha=.5, color="navy")
    ax_hist.set_xlabel(f"{prefix}Curvature")
    ax_hist.set_yticks([], [])

    ax_hist.spines[['right', 'top', 'left']].set_visible(False)

    if output_path_2 is not None:
        plt.savefig(output_path_2, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    # write mesh to tensorboard
    if input_dim == 3 and writer is not None:
        if writer is not None:
            # prepare for tensorboard mesh
            log_curvature_rgb = cmap_labels(minmax(scaled_curvature[middle_idx]))
            log_curvature_rgb = torch.unsqueeze(log_curvature_rgb, 0)
            outputs_unsqueezed = torch.unsqueeze(outputs[middle_idx], 0)

            point_size_config = {
                #    'material': {
                #        'cls': 'PointsMaterial',
                #        'size': .2
                #    }
            }

            writer.add_mesh("curvature",
                            vertices=outputs_unsqueezed,
                            colors=log_curvature_rgb,
                            config_dict=point_size_config)

    # write to tensorboard
    if writer is not None:
        writer.add_figure("curvature/colorcode", fig_col)
        writer.add_figure("curvature/histogram", fig_hist)
        if input_dim == 3:
            writer.add_figure("curvature/colorcode/3d", fig_col_3d)

    # add summary
    ax_col.remove()
    ax_hist.remove()

    fig_sum = join_plots([[(ax_col, scatter_col), ax_hist]],
                         latent_activations=latent_activations,
                         labels=labels,
                         title=f"{prefix}Curvature")

    if writer is not None:
        writer.add_figure("curvature/summary", fig_sum)

        # clean up tensorboard writer
        writer.flush()
        writer.close()


def plot_determinants(model,
                      dataloader,
                      quantile=1.,
                      batch_size=-1,
                      scaling="asinh",
                      grid="dataset",
                      num_steps=None,
                      device="cpu",
                      output_path_1=None,
                      output_path_2=None,
                      writer=None,
                      x_lim_hist=None):
    print("[Analyse] determinants ...")

    # forward pass
    inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    # initialize diffgeo objects
    pbm = PullbackMetric(2, model.decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    # calculate coordinates
    coordinates = get_coordinates(latent_activations, grid=grid, num_steps=num_steps).to(device)

    # batch-size is negative use the whole batch, i.e. don't batch
    if batch_size == -1:
        batch_size = coordinates.shape[0]

    # calculate determinants
    determinants = None
    for coord in torch.utils.data.DataLoader(coordinates, batch_size=batch_size):
        batch_dets = rm.metric_det(base_point=coord).detach()

        if determinants is None:
            determinants = batch_dets
        else:
            determinants = torch.hstack((determinants, batch_dets))

    # scale determinants
    scaling_fn, prefix = determine_scaling_fn(scaling)
    dets_scaled = scaling_fn(determinants)

    # only consider quantile
    middle_idx = values_in_quantile(dets_scaled, quantile)

    """
    PLOTTING
    """

    coordinates = coordinates.detach().cpu()
    dets_scaled = dets_scaled.detach().cpu()
    determinants = determinants.detach().cpu()
    latent_activations = latent_activations.detach().cpu()

    # plot color-coded determinants
    fig_col, ax_col = plt.subplots(figsize=((5, 5)))

    # fig_col.suptitle(f"{prefix}Determinants")

    scatter_col = ax_col.scatter(coordinates[:, 0][middle_idx],
                                 coordinates[:, 1][middle_idx],
                                 c=dets_scaled[middle_idx],
                                 **get_sc_kwargs())
    divider = make_axes_locatable(ax_col)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    sm = ScalarMappable()
    sm.set_array(determinants)

    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_alpha(0.5)
    cbar.draw_all()
    ax_col.set_aspect("equal")
    ax_col.axis("off")

    if output_path_1 is not None:
        plt.savefig(output_path_1, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    # plot histogram of the determinants
    fig_hist, ax_hist = plt.subplots(figsize=(5, 5))
    # fig_hist.suptitle(f"{prefix}Determinants Distribution")
    ax_hist.hist(dets_scaled.numpy(), bins=40, density=True, alpha=.5, color="navy")
    ax_hist.set_xlabel(rf"{prefix}determinant")

    ax_hist.spines[['right', 'top', 'left']].set_visible(False)
    ax_hist.set_yticks([], [])

    if x_lim_hist:
        ax_hist.set_xlim(*x_lim_hist)

    if output_path_2 is not None:
        plt.savefig(output_path_2, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    if writer is not None:
        writer.add_figure("determinants/colorcode", fig_col)
        writer.add_figure("determinants/histogram", fig_hist)

    ax_hist.remove()
    ax_col.remove()

    fig_sum = join_plots([[(ax_col, scatter_col), ax_hist]],
                         latent_activations=latent_activations,
                         labels=labels,
                         title=f"{prefix}Determinants")

    if writer is not None:
        writer.add_figure("determinants/summary", fig_sum)

        # clean up tensorboard writer
        writer.flush()
        writer.close()


def indicatrices(model,
                 dataloader,
                 grid="convex_hull",
                 device="cpu",
                 num_steps=20,
                 num_gon=50,
                 output_path=None,
                 writer=None):
    print("[Analysis] Indicatrices...")

    inputs, outputs, latent_activations, labels = data_forward(model, dataloader)

    coordinates = get_coordinates(latent_activations.detach().cpu(), num_steps=num_steps, grid=grid).to(device)

    # calculate grid distance
    xrange = (torch.max(latent_activations[:, 0]).item() - torch.min(latent_activations[:, 0])).item()
    yrange = (torch.max(latent_activations[:, 1]).item() - torch.min(latent_activations[:, 1])).item()
    distance_grid = min(xrange / (num_steps - 1), yrange / (num_steps - 1)) * 2

    # initialize diffgeo objects
    pbm = PullbackMetric(2, model.decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    metric_matrices = rm.metric.metric_matrix(coordinates)

    # generate vector patches at grid points, normed in pullback metric
    vector_patches = rm.generate_unit_vectors(num_gon, coordinates).to(device)

    # scale vectors and attach them to their corresponding tangent spaces
    max_vector_norm = torch.max(torch.linalg.norm(vector_patches, dim=1)).item()
    max_vector_norm = min(max_vector_norm, UPPER_EPSILON)
    normed_vector_patches = vector_patches / max_vector_norm * distance_grid
    anchored_vector_patches = coordinates.unsqueeze(1).expand(*normed_vector_patches.shape) + normed_vector_patches

    # create polygons
    polygons = [Polygon(tuple(vector.tolist()), True) for vector in anchored_vector_patches]

    """
    Plotting
    """
    latent_activations = latent_activations.detach().cpu()

    # plot blobs
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(latent_activations[:, 0], latent_activations[:, 1], c=labels, cmap="tab10", **get_sc_kwargs())
    p = PatchCollection(polygons)
    p.set_color([156 / 255, 0, 255 / 255])

    ax.add_collection(p)
    ax.set_aspect("equal")
    ax.axis("off")
    # fig.suptitle(f"Indicatrices")

    if output_path is not None:
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

    if writer is not None:
        writer.add_figure("indicatrix", fig)

        # clean up tensorboard writer
        writer.flush()
        writer.close()
