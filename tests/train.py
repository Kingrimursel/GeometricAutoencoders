import time
import torch
from diffgeo.connections import LeviCivitaConnection
from diffgeo.manifolds import RiemannianManifold
from diffgeo.metrics import PullbackMetric
from matplotlib import cm

from torch import optim, nn
from conf import device
from criterions import DistanceLoss, DeterminantLoss, CurvatureLoss, ChristoffelLoss, IndicatrixLoss
from torch.nn.utils import clip_grad_norm_
from util import Color, Animator, get_rg_value
from AutoEncoderVisualization.data.handle_data import data_forward


def train_model(model,
                train_loader,
                model_path,
                epochs=50,
                alpha=1e-5,
                beta=1e-5,
                gamma=1e-5,
                delta=1e-5,
                epsilon=1e-5,
                n_dist_samples=1,
                weight_decay=0.,
                std=None,
                n_gaussian_samples=None,
                n_origin_samples=None,
                writer=None,
                mode="normal",
                create_video=False
                ):
    """
    Train the model
    :param model: instance of the model
    :param train_loader: dataloader of test dataset
    :param model_path: the path for model weights
    :param epochs: number of epochs to train for
    :param alpha: rate for determinant loss
    :param beta: rate for distance loss
    :param delta: rate for curvature loss
    :param n_dist_samples: the number of samples to compute the distances for
    :param weight_decay: weight decay
    :param writer: SummaryWriter instance
    :param save_model: whether the model should be saved to file
    :param std: the standard deviation of the gaussians for the curvature loss
    :return:
    """

    # initialize model
    # create optimizer object
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    distance_criterion = DistanceLoss(model=model,
                                      n_dist_samples=n_dist_samples)

    determinant_criterion = DeterminantLoss(model=model)

    curvature_criterion = CurvatureLoss(model=model,
                                        n_gaussian_samples=n_gaussian_samples,
                                        n_origin_samples=n_origin_samples,
                                        std=std)

    christoffel_criterion = ChristoffelLoss(model=model)

    indicatrix_criterion = IndicatrixLoss(model=model)

    if create_video:
        if model.input_dim == 784:
            animator1 = Animator()
            animator2 = Animator()

            pbm = PullbackMetric(2, model.decoder)
            lcc = LeviCivitaConnection(2, pbm)
            rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)
            cmap_tab10 = cm.get_cmap("tab10", 12)
            cmap_viridis = cm.get_cmap("viridis", 12)
        else:
            input, _, _, _ = data_forward(model, train_loader)

            animator1 = Animator()
            animator2 = Animator(dim=3, ground_truth=input.cpu().detach()[::40])

    # whether to log each epoch or each batch
    log_epoch = False
    counter = 0
    for epoch in range(epochs):
        print(f"{Color.GREEN}epoch {epoch + 1} of {epochs}{Color.NC}")

        epoch_total_loss = 0
        epoch_mse_loss = 0
        epoch_det_loss = 0
        epoch_dist_loss = 0
        epoch_curv_loss = 0

        for i, (batch_features, _) in enumerate(train_loader):
            # print(f"batch {i + 1} of {len(train_loader)}")
            batch_features = batch_features.view(-1, model.input_dim).to(device)

            # forward pass
            outputs = model(batch_features)

            # zero the gradients
            model.zero_grad()

            # calculate the losses
            mse_loss = nn.MSELoss()(outputs, batch_features)

            # the custom losses
            if not mode == "vanilla":
                if alpha != 0. or mode == "baseline":
                    det_loss = determinant_criterion(epoch=epoch)
                else:
                    det_loss = torch.tensor([0.], device=device)

                if beta != 0. or mode == "baseline":
                    dist_loss = distance_criterion(outputs, epoch=epoch)
                else:
                    dist_loss = torch.tensor([0.], device=device)

                if delta != 0. or mode == "baseline":
                    curv_loss = torch.tensor([0.], device=device)
                    christoffel_loss = torch.tensor([0.], device=device)

                    # curv_loss = curvature_criterion(epoch=epoch)
                    # christoffel_loss = christoffel_criterion(epoch=epoch)
                else:
                    curv_loss = torch.tensor([0.], device=device)
                    christoffel_loss = torch.tensor([0.], device=device)

                if gamma != 0. or mode == "baseline":
                    indicatrix_loss = indicatrix_criterion(epoch=epoch)
                else:
                    indicatrix_loss = torch.tensor([0.], device=device)
            else:
                dist_loss = torch.tensor([0.], device=device)
                det_loss = torch.tensor([0.], device=device)
                curv_loss = torch.tensor([0.], device=device)
                christoffel_loss = torch.tensor([0.], device=device)
                indicatrix_loss = torch.tensor([0.], device=device)

            if not mode == "baseline":
                total_loss = mse_loss + alpha * det_loss + beta * dist_loss + delta * curv_loss + epsilon * christoffel_loss + gamma * indicatrix_loss
            else:
                total_loss = mse_loss

            total_loss.backward()

            # for name, param in model.named_parameters():
            #    if torch.max(param.grad) > 10:
            #        print(name, torch.max(param.grad))
            # import sys
            # sys.exit()
            # test = clip_grad_norm_(model.parameters(), max_norm=1, error_if_nonfinite=True)

            optimizer.step()

            # print(f"forward: {t_end_forward - t_start_forward}")
            # print(f"backward: {t_end_backward - t_start_backward}")
            # log to tensorboard
            if not log_epoch:
                writer.add_scalar("total loss", total_loss, counter)
                writer.add_scalar("mse loss", mse_loss, counter)
                writer.add_scalar("det loss", det_loss, counter)
                writer.add_scalar("distance loss", dist_loss, counter)
                writer.add_scalar("curvature loss", curv_loss, counter)
                writer.add_scalar("christoffel loss", christoffel_loss, counter)
                writer.add_scalar("indicatrix loss", indicatrix_loss, counter)

            counter += 1

            if create_video:
                # append to animations
                if i % 10 == 0:  # better do 10!! 25, and 30 fps
                    _, outputs, latent_activations, labels = data_forward(model, train_loader)
                    latent_activations_copied = latent_activations.cpu().detach()
                    if model.input_dim == 784:

                        determinants = rm.metric_det(base_point=latent_activations).detach()
                        determinants = torch.asinh(determinants).cpu()

                        labels_color = torch.tensor(cmap_tab10(labels)[:, :-1])
                        determinants_color = torch.tensor(cmap_viridis(determinants)[:, :-1])

                        animator1.append(latent_activations_copied, colors=labels_color)
                        animator2.append(latent_activations_copied, colors=determinants_color)
                    else:
                        rgb_pos = get_rg_value(latent_activations_copied)
                        animator1.append(latent_activations_copied, colors=rgb_pos)
                        animator2.append(outputs.cpu().detach(), colors=rgb_pos)

        if log_epoch:
            writer.add_scalar("total loss", epoch_total_loss, epoch)
            writer.add_scalar("mse loss", epoch_mse_loss, epoch)
            writer.add_scalar("det loss", epoch_det_loss, epoch)
            writer.add_scalar("distance loss", epoch_dist_loss, epoch)
            writer.add_scalar("curvature loss", epoch_curv_loss, epoch)

    if create_video:
        # evaluate animator
        animator1.evaluate()
        animator1.animate("/export/home/pnazari/workspace/AutoEncoderVisualization/stuff/animation1_spheres_new.mp4",
                          fps=30,  # 30
                          legend=True,
                          figsize=(6, 6),
                          s=.5,
                          dpi=500)  # 500

        animator2.evaluate()
        animator2.animate("/export/home/pnazari/workspace/AutoEncoderVisualization/stuff/animation2_spheres_new.mp4",
                          fps=30,  # 30
                          figsize=(6, 6),
                          legend=True,
                          s=.5,
                          dpi=500)  # 500

    if model_path is not None:
        # save trained model to file
        print("[model] saving...")
        torch.save(model.state_dict(), model_path)
    else:
        print("[model] not saving...")

    # clean up tensorboard writer
    writer.flush()
    writer.close()

    return mse_loss
