import torch
import torchvision

from conf import device
from data.custom import SwissRoll, Spheres, Chiocciola, Saddle, HyperbolicParabloid, Earth, FigureEight, Zilionis

from data.custom import WoollyMammoth


def load_data(train_batch_size=128, test_batch_size=32, dataset="MNIST"):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    if dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(root="/export/ial-nfs/user/pnazari/data",
                                                   train=True,
                                                   transform=transform,
                                                   download=True)

        test_dataset = torchvision.datasets.MNIST(root="/export/ial-nfs/user/pnazari/data",
                                                  train=False,
                                                  transform=transform,
                                                  download=True)
    elif dataset == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(root="/export/ial-nfs/user/pnazari/data",
                                                          train=True,
                                                          transform=transform,
                                                          download=True)

        test_dataset = torchvision.datasets.FashionMNIST(root="/export/ial-nfs/user/pnazari/data",
                                                         train=False,
                                                         transform=transform,
                                                         download=True)
    elif dataset == "SwissRoll":
        train_dataset = SwissRoll(n_samples=100000)
        test_dataset = SwissRoll(n_samples=100000)
    elif dataset == "chiocciola":
        train_dataset = Chiocciola(n_samples=100000)
        test_dataset = Chiocciola(n_samples=10000)
    elif dataset == "Mammoth":
        train_dataset = WoollyMammoth()
        test_dataset = WoollyMammoth()
    elif dataset == "Spheres":
        train_dataset = Spheres(n_samples=100000, n_spheres=2, noise=0.)
        test_dataset = Spheres(n_samples=10000, n_spheres=2, noise=0.)
    elif dataset == "Saddle":
        train_dataset = Saddle(n_samples=100000)
        test_dataset = Saddle(n_samples=100000)
    elif dataset == "HyperbolicParabloid":
        test_dataset = HyperbolicParabloid(n_samples=100000)
        train_dataset = HyperbolicParabloid(n_samples=100000)
    elif dataset == "Earth":
        train_dataset = Earth(n_samples=100000,
                              filename="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/earth/landmass.pt")
        test_dataset = Earth(n_samples=10000,
                             filename="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/earth/landmass.pt")
    elif dataset == "FigureEight":
        train_dataset = FigureEight(n_samples=100000)
        test_dataset = FigureEight(n_samples=5000)
    elif dataset == "Zilionis":
        train_dataset = Zilionis(dir_path="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/zilionis")
        test_dataset = Zilionis(dir_path="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/zilionis")
    else:
        return

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, num_workers=4)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                               num_workers=16)  # pin_memory=True

    return train_loader, test_loader


# TODO: move into base class of autoencoders
def data_forward(model, test_loader):
    inputs = torch.tensor([])
    outputs = torch.tensor([])
    latent_activations = torch.tensor([])
    labels = torch.tensor([])

    for k, (batch_features, batch_labels) in enumerate(test_loader):
        # do that in order to get activations at latent layer
        batch_features = batch_features.view(-1, model.input_dim).to(device)
        output = model(batch_features)

        if k == 0:
            inputs = batch_features
            outputs = output
            latent_activations = model.latent_activations
            labels = batch_labels
        else:
            inputs = torch.vstack((inputs, batch_features))
            outputs = torch.vstack((outputs, output))
            latent_activations = torch.vstack((latent_activations, model.latent_activations))
            labels = torch.hstack((labels, batch_labels))

    return inputs, outputs, latent_activations, labels
