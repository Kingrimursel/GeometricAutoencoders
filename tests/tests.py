import json
import os
import sys

import numpy as np
import pandas as pd
import torchvision
import umap
from openTSNE import TSNE
from functorch import jacrev, jacfwd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse, Polygon
from torch import optim, nn
from torch.autograd.functional import jacobian

from conf import device, UPPER_EPSILON
from data.custom import SwissRoll, Saddle, Earth, FigureEight, WoollyMammoth, Spheres, Zilionis

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import torch
from matplotlib import pyplot as plt
from pathlib import Path

from diffgeo.manifolds import RiemannianManifold
from diffgeo.metrics import PullbackMetric
from diffgeo.connections import LeviCivitaConnection

from util import Color, batch_jacobian, symlog, values_in_quantile, get_coordinates, transform_axes, get_sc_kwargs

encoder_name = "sqrt"

input_dim = 784


def encoder(x):
    # multiplier = torch.tensor([[x[0] ** (-1 / 2), 0], [0, x[1] ** (-2 / 3)]])
    # multiplier = torch.tensor([[-1., 1.], [1., 0.]])

    # output = multiplier @ x

    return torch.pow(x, 1 / 2)  # torch.pow(x, 1 / 2)


def decoder(x):
    # multiplier = torch.tensor([[x[0] ** (2 - 1), 0], [0, x[1] ** (3 - 1)]])
    # multiplier = torch.tensor([[0., 1.], [x[0], 1.]])
    # output = multiplier @ x

    # multiplier = torch.tensor([[x[0], 0.], [0., x[1]]], device=device)
    # output = multiplier @ x

    # try:
    #    return (x.T @ torch.diag(torch.sum(x, dim=1))).T
    # except:
    #    return torch.sum(x) * x * x[0]

    # stereographic projection

    X = 2 * x[0] / (1 + x[0] ** 2 + x[1] ** 2)
    Y = 2 * x[1] / (1 + x[0] ** 2 + x[1] ** 2)
    Z = (-1 + x[0] ** 2 + x[1] ** 2) / (1 + x[0] ** 2 + x[1] ** 2)

    res = torch.vstack((X, Y, Z))

    return res


"""
DIFFERENT TESTS
"""


def umap_zilionis():
    zilionis = Zilionis(dir_path="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/zilionis")

    seed = 0

    umapperns_after = umap.UMAP(metric="cosine",
                                n_neighbors=30,
                                n_epochs=750,
                                # log_losses="after",
                                random_state=seed,
                                verbose=True)
    latent_activations = umapperns_after.fit_transform(zilionis.dataset)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis("off")

    scatter = ax.scatter(latent_activations[:, 0],
                         latent_activations[:, 1],
                         **get_sc_kwargs(),
                         c=zilionis.labels,
                         cmap="tab10")

    output_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/stuff/zilionis_umap.png"

    plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

def test_zilionis():
    z = Zilionis(dir_path="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/zilionis")

    print(torch.max(torch.abs(torch.mean(z.dataset, dim=1))))
    print(torch.std(z.dataset, dim=1))

    # print(z.dataset.shape)
    # print(torch.min(z.dataset, dim=1).values.shape, torch.max(z.dataset, dim=1))


def tsne_umap():
    # mammoth = Earth(n_samples=10000,
    #                filename="/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/earth/landmass.pt")

    # mammoth = WoollyMammoth()

    mammoth = Spheres(n_samples=10000, n_spheres=2, noise=0.)

    dataset = mammoth.dataset
    labels = mammoth.labels

    # latent_activations = umap.UMAP(n_neighbors=200, min_dist=0.1).fit_transform(dataset)
    latent_activations = TSNE().fit(dataset)

    # fig.suptitle("Latent Space")

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis("off")

    scatter = ax.scatter(latent_activations[:, 0],
                         latent_activations[:, 1],
                         **get_sc_kwargs(),
                         c=labels,
                         cmap="tab10")

    output_path = "/export/home/pnazari/workspace/AutoEncoderVisualization/stuff/spheres_tsne.png"

    plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()


def test_stereo():
    x = torch.linspace(-1, 1, 20)
    y = torch.linspace(-1, 1, 20)

    X, Y = torch.meshgrid(x, y, indexing="ij")

    coordinates = torch.vstack([X.ravel(), Y.ravel()]).T.to(device)

    pbm = PullbackMetric(2, decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    curvature = rm.sectional_curvature(torch.tensor([1., 0.], device=device),
                                       torch.tensor([0., 1.], device=device),
                                       coordinates)

    print(curvature)


def test_figure_eight():
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib as mpl

    fig = plt.figure()
    ax = plt.axes()

    f8 = FigureEight(1000)

    i = 100

    ax.scatter(*f8.dataset.T, c=f8.labels)

    plt.show()


def test_world():
    # import geopandas

    # file = "/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/earth/ne_10m_coastline.shp"
    # world = geopandas.read_file(file)
    # print(world.columns)
    # world.plot()p
    # plt.show()

    earth = Earth(n_samples=100 * 100,
                  filename=f"/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/earth/landmass.pt")

    result = earth.generate(2000)
    dataset, labels = earth.create()

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # transform_axes(ax)

    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=labels, marker=".", s=1)

    ax.view_init(azim=90)

    plt.show()

    """ 
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import geopandas
    from sklearn import preprocessing

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    bm = Basemap(projection="cyl")

    xs = []
    ys = []
    zs = []

    phis = []
    thetas = []

    # phi = long, theta = lat
    # das erste Argument is azimuth (phi), das zweite polar (theta) (in [-pi, pi])
    for phi in np.linspace(-180, 180, num=100):
        for theta in np.linspace(-90, 90, num=100):
            if bm.is_land(phi, theta):
                phis.append(phi)
                thetas.append(theta)

                phi_rad = phi / 360 * 2 * np.pi
                theta_rad = theta / 360 * 2 * np.pi

                x = np.sin(theta_rad) * np.cos(phi_rad)
                y = np.sin(theta_rad) * np.sin(phi_rad)
                z = np.cos(theta_rad)

                xs.append(x)
                ys.append(y)
                zs.append(z)

    # generate labels
    df = pd.DataFrame(
        {
            "longitude": phis,
            "latitude": thetas
        }
    )

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))

    results = geopandas.sjoin(gdf, world, how="left")

    le = preprocessing.LabelEncoder()
    encoded_results = torch.tensor(le.fit_transform(results["continent"].values))



    # print(torch.tensor(results["continent"].values))

    ax.scatter(xs, ys, zs, marker=".", s=1)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    zs = torch.tensor(zs)

    coordinates = torch.vstack((xs, ys, zs, encoded_results)).T

    torch.save(coordinates, "/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/earth/landmass.pt")

    print(coordinates.shape)

    plt.show()
    """


def test_quant():
    t = torch.linspace(-10, 10, steps=21)
    # t = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    tt = values_in_quantile(t, q=0.7)
    tt = t[tt]
    print(t)
    print(tt)
    pass


def test_set():
    test = Saddle(n_samples=10000)

    data = test.dataset
    labels = test.labels

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)

    plt.show()

    pass


def test_dict():
    s = torch.tensor([-100., -10., -1., -0.1, 0., 0.1, 1., 10., 100.])
    t = symlog(s)
    print(t)


def batch_jac():
    points = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])

    def test(x):
        try:
            return (x.T @ torch.diag(torch.sum(x, dim=1))).T
        except:
            return torch.sum(x) * x

    jac = batch_jacobian(test, points)

    test = jac[:, :, 0]
    # print(jac.shape)
    print(jac)
    print(test)

    pass


def batch_inv():
    T = torch.tensor([[[1., 0.], [0., 1.]], [[1., 1.], [0., 0.]]], device=device)
    test = torch.linalg.inv_ex(T)
    print(test.inverse[test.info != 0.])
    test.inverse[test.info != 0.] = torch.zeros((2, 2), device=device)
    print(test)
    pass


def indic():
    pbm = PullbackMetric(2, decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    num_steps = 5

    latent_activations = torch.tensor([[0., 2.], [2., 0.]], device=device)

    coordinates = get_coordinates(latent_activations, num_steps=5, grid="min_square").to(device)
    # calculate grid distance
    xrange = (torch.max(latent_activations[:, 0]).item() - torch.min(latent_activations[:, 0])).item()
    yrange = (torch.max(latent_activations[:, 1]).item() - torch.min(latent_activations[:, 1])).item()
    distance_grid = min(xrange / (num_steps - 1), yrange / (num_steps - 1)) * 2

    metric_matrices = rm.metric.metric_matrix(coordinates)

    U, S, Vh = torch.svd(metric_matrices)

    S /= 100

    for u in U:
        test = torch.arccos(u[0, 0] / torch.norm(u[:, 0])) * 360 / 2 / 3.1415

    ellipses = [Ellipse(center, eigenvalues[0], eigenvalues[1],
                        torch.arccos(u[0, 0] / torch.norm(u[:, 0])) * 360 / 2 / 3.1415) for center, eigenvalues, u in
                zip(coordinates, S, U)]

    # generate vector patches at grid points, normed in pullback metric
    vector_patches = rm.generate_unit_vectors(50, coordinates).to(device)

    # scale vectors and attach them to their corresponding tangent spaces
    max_vector_norm = torch.max(torch.linalg.norm(vector_patches, dim=1, ord=2)).item()
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
    fig, ax = plt.subplots()

    p = PatchCollection(polygons)
    p.set_color([156 / 255, 0, 255 / 255])

    ax.add_collection(p)
    ax.set_aspect("equal")
    fig.suptitle(f"Indicatrices")

    p2 = PatchCollection(ellipses)
    # ax.add_collection(p2)

    plt.show()


def batch_curv():
    curvatures = []
    christoffels = []
    metric_matrices = []
    riemannian_curvatures = []
    riemannian_curvature_tensors = []
    christoffel_derivatives = []
    term_1s = []
    term_2s = []
    term_3s = []
    term_4s = []

    for alpha in torch.arange(1, 10):
        print(alpha.item())

        def decoder(x):
            # multiplier = torch.tensor([[x[0] ** (2 - 1), 0], [0, x[1] ** (3 - 1)]])
            # multiplier = torch.tensor([[0., 1.], [x[0], 1.]])
            # output = multiplier @ x

            # multiplier = torch.tensor([[x[0], 0], [0, x[1] ** 2]])
            # output = multiplier @ x

            x = 1 / alpha * x

            try:
                return (x.T @ torch.diag(torch.sum(x, dim=1))).T
            except:
                return torch.sum(x) * x * x[0]

        pbm = PullbackMetric(2, decoder)
        lcc = LeviCivitaConnection(2, pbm)
        rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

        coordinates = alpha * torch.tensor([[2e-3, 1e-3]], device=device)
        num_coordinates = coordinates.shape[0]

        # curvature = rm.sectional_curvature(torch.tensor([1., 0.], device=device),
        #                                   torch.tensor([0., 1.], device=device),
        #                                   coordinates)

        metric_matrix = rm.metric.metric_matrix(coordinates)

        # riemannian_curvature = rm.riemannian_curvature(torch.tensor([1., 0.], device=device),
        #                                               torch.tensor([0., 1.], device=device),
        #                                               torch.tensor([1., 0.], device=device), coordinates)

        riemannian_curvature_tensor, term_1, term_2, term_3, term_4 = rm.riemannian_curvature_tensor(coordinates)

        christoffel = rm.metric.christoffels(coordinates)

        christoffel_derivative = rm.christoffel_derivative(coordinates)

        # riemannian_curvatures.append(riemannian_curvature)
        riemannian_curvature_tensors.append(riemannian_curvature_tensor)
        term_1s.append(term_1)
        term_2s.append(term_2)
        term_3s.append(term_3)
        term_4s.append(term_4)

        metric_matrices.append(metric_matrix)
        christoffels.append(christoffel)
        christoffel_derivatives.append(christoffel_derivative)
        # curvatures.append(curvature)

    # curvatures = torch.cat(curvatures)
    # metric_matrices = torch.cat(metric_matrices)
    # riemannian_curvatures = torch.cat(riemannian_curvatures)
    # christoffels = torch.cat(christoffels)
    riemannian_curvature_tensors = torch.cat(riemannian_curvature_tensors)
    term_1s = torch.cat(term_1s)
    term_2s = torch.cat(term_2s)
    term_3s = torch.cat(term_3s)
    term_4s = torch.cat(term_4s)

    # christoffel_derivatives = torch.cat(christoffel_derivatives)

    # print(christoffels / christoffels[0])
    # print(christoffel_derivatives / christoffel_derivatives[0])
    # print(torch.sum((christoffels / christoffels[0]) != (christoffel_derivatives / christoffel_derivatives[0])))

    test = term_1s + term_2s + term_3s + term_4s

    # print(torch.where(test == -0.0625))

    torch.set_printoptions(precision=8)
    # print(term_1s / term_1s[0])
    # print(term_2s / term_2s[0])
    # print(term_3s / term_3s[0])
    # print(term_4s / term_4s[0])
    # print(test)
    # print(test[0])
    # print(test / test[0])

    # print(riemannian_curvature_tensors / riemannian_curvature_tensors[0])

    # print(riemannian_curvature_tensors / riemannian_curvature_tensors[0])
    # print(christoffels / christoffels[0])
    # print(riemannian_curvature_tensors[0] / riemannian_curvature_tensors)

    # print(tts[0] / tts)
    # print(cs[0] / cs)
    # print(ts[0] / ts)

    # test = rm.sectional_curvature(torch.tensor([1., 0.], device=device),
    #                              torch.tensor([0., 1.], device=device),
    #                              coordinates)

    # print(curvature)
    # print(curvature)
    # print(test)


def mam():
    file = "/data/raw/mammoth/mammoth_3d.json"

    ufile = "/export/home/pnazari/workspace/AutoEncoderVisualization/data/raw/mammoth_umap.json"

    uf = open(ufile)
    udata = json.load(uf)
    uf.close()

    print(udata.keys())

    f = open(file)
    data = torch.tensor(json.load(f))
    f.close()

    data = torch.tensor(udata["3d"])

    print(torch.tensor(data).shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    a = 0
    b = 10000
    ax.scatter(data[:, 0][a:b], data[:, 1][a:b], data[:, 2][a:b], s=1, c=udata["labels"])
    plt.show()


def test_swiss_roll():
    from sklearn.datasets import make_swiss_roll
    sr = SwissRoll(n_samples=5000, n_rolls=1)
    test, _ = make_swiss_roll(n_samples=5000)

    print(test.shape)
    print(sr.dataset.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(sr.dataset[:, 0, 0], sr.dataset[:, 1, 0], sr.dataset[:, 2, 0])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(sr.dataset[:, 0, 1], sr.dataset[:, 1, 1], sr.dataset[:, 2, 1])
    plt.show()


def test_n2n_grad():
    # initialize diffgeo objects
    pbm = PullbackMetric(2, decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    test = torch.zeros((3, input_dim))

    test[0, 1] = float("nan")

    print(rm.metric.metric_det(test))


def test_curvature():
    # TODO: is this dependant of the pair of vectors I chose?
    # create output directory if not exists
    Path(f"/stuff/old/tests/curvature/{encoder_name}/").mkdir(
        parents=True,
        exist_ok=True)

    # initialize diffgeo objects
    pbm = PullbackMetric(2, decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    x_min = 1
    x_max = 3
    y_min = 1
    y_max = 3

    grid_steps = 50

    # generate coordinate grid
    grid_x, grid_y = torch.meshgrid(torch.linspace(x_min, x_max, steps=grid_steps),
                                    torch.linspace(y_min, y_max, steps=grid_steps),
                                    indexing="ij")

    # prepare coordinate tuples
    coordinates = torch.vstack([grid_x.ravel(), grid_y.ravel()]).T
    num_coordinates = coordinates.shape[0]

    def metric(x, y):
        M = torch.tensor([[4 * x ** 2 * y ** 2 + y ** 4, 2 * x ** 3 * y + 2 * x * y ** 3],
                          [2 * x * y ** 3 + 2 * y * x ** 3, x ** 4 + 4 * x ** 2 * y ** 2]])
        return M

    def invmetric(x, y):
        return torch.tensor(
            [[(x ** 2 + 4 * y ** 2) / (9 * x ** 2 * y ** 4), (-2 * x ** 2 - 2 * y ** 2) / (9 * x ** 3 * y ** 3)],
             [(-2 * x ** 2 - 2 * y ** 2) / (9 * x ** 3 * y ** 3), (4 * x ** 2 + y ** 2) / (9 * x ** 4 * y ** 2)]])

    def christ_xxx(base_point):
        t1 = (base_point[0] ** 2 + 4 * base_point[1] ** 2) / (9 * base_point[0] ** 2 * base_point[1] ** 4) * 8 * \
             base_point[0] * base_point[1] ** 2
        t2 = (-2 * base_point[0] ** 2 - 2 * base_point[1] ** 2) / (9 * base_point[0] ** 3 * base_point[1] ** 3) * (
                4 * base_point[1] ** 3 + 12 * base_point[0] ** 2 * base_point[1] - 8 * base_point[0] ** 2 * base_point[
            1] - 4 * base_point[1] ** 3)
        g = 1 / 2 * (t1 + t2)
        return g

    def christ_yxx(base_point):
        t1 = (-2 * base_point[0] ** 2 - 2 * base_point[1] ** 2) / (9 * base_point[0] ** 3 * base_point[1] ** 3) * 8 * \
             base_point[0] * base_point[1] ** 2
        t2 = (4 * base_point[0] ** 2 + base_point[1] ** 2) / (9 * base_point[0] ** 4 * base_point[1] ** 2) * (
                4 * base_point[1] ** 3 + 12 * base_point[0] ** 2 * base_point[1] - 8 * base_point[0] ** 2 * base_point[
            1] - 4 * base_point[1] ** 3)
        g = 1 / 2 * (t1 + t2)
        return g

    def christ_xxy(base_point):
        t1 = (-2 * base_point[0] ** 2 - 2 * base_point[1] ** 2) / (9 * base_point[0] ** 3 * base_point[1] ** 3) * (
                4 * base_point[0] ** 3 + 8 * base_point[0] * base_point[1] ** 2)
        t2 = (base_point[0] ** 2 + 4 * base_point[1] ** 2) / (9 * base_point[0] ** 2 * base_point[1] ** 4) * (
                8 * base_point[0] ** 2 * base_point[1] + 4 * base_point[1] ** 3)
        g = 1 / 2 * (t1 + t2)
        return g

    christ_xyx = christ_xxy

    def christ_yyx(base_point):
        t1 = (-2 * base_point[0] ** 2 - 2 * base_point[1] ** 2) / (9 * base_point[0] ** 3 * base_point[1] ** 3) * (
                8 * base_point[0] ** 2 * base_point[1] + 4 * base_point[1] ** 3)
        t2 = (4 * base_point[0] ** 2 + base_point[1] ** 2) / (9 * base_point[0] ** 4 * base_point[1] ** 2) * (
                6 * base_point[0] * base_point[1] ** 2 + 2 * base_point[0] ** 3 + 4 * base_point[0] ** 3 + 8 *
                base_point[0] * base_point[1] ** 2 - 6 * base_point[0] * base_point[1] ** 2 - 2 * base_point[0] ** 3)
        g = 1 / 2 * (t1 + t2)
        return g

    christ_yxy = christ_yyx

    def christ_yyy(base_point):
        t1 = (-2 * base_point[0] ** 2 - 2 * base_point[1] ** 2) / (9 * base_point[0] ** 3 * base_point[1] ** 3) * (
                12 * base_point[0] * base_point[1] ** 2 - 8 * base_point[0] * base_point[1] ** 2)
        t2 = (4 * base_point[0] ** 2 + base_point[1] ** 2) / (9 * base_point[0] ** 4 * base_point[1] ** 2) * (
                8 * base_point[0] ** 2 * base_point[1])
        g = 1 / 2 * (t1 + t2)
        return g

    def christ_xyy(base_point):
        t1 = (base_point[0] ** 2 + 4 * base_point[1] ** 2) / (9 * base_point[0] ** 2 * base_point[1] ** 4) * (
                4 * base_point[0] ** 3 + 12 * base_point[0] * base_point[1] ** 2 - 4 * base_point[0] ** 3 - 8 *
                base_point[0] * base_point[1] ** 2)
        t2 = (-2 * base_point[0] ** 2 - 2 * base_point[1] ** 2) / (9 * base_point[0] ** 3 * base_point[1] ** 3) * 8 * \
             base_point[0] ** 2 * base_point[1]
        g = 1 / 2 * (t1 + t2)
        return g

    def hand_christoffels(base_point):
        c1 = torch.tensor([[[1, 0],
                            [0, 0]],

                           [[0, 0],
                            [0, 0]]]) * christ_xxx(base_point)

        c2 = torch.tensor([[[0, 1],
                            [0, 0]],

                           [[0, 0],
                            [0, 0]]]) * christ_xxy(base_point)

        c3 = torch.tensor([[[0, 0],
                            [1, 0]],

                           [[0, 0],
                            [0, 0]]]) * christ_xyx(base_point)

        c4 = torch.tensor([[[0, 0],
                            [0, 1]],

                           [[0, 0],
                            [0, 0]]]) * christ_xyy(base_point)
        c5 = torch.tensor([[[0, 0],
                            [0, 0]],

                           [[1, 0],
                            [0, 0]]]) * christ_yxx(base_point)

        c6 = torch.tensor([[[0, 0],
                            [0, 0]],

                           [[0, 1],
                            [0, 0]]]) * christ_yxy(base_point)
        c7 = torch.tensor([[[0, 0],
                            [0, 0]],

                           [[0, 0],
                            [1, 0]]]) * christ_yyx(base_point)
        c8 = torch.tensor([[[0, 0],
                            [0, 0]],

                           [[0, 0],
                            [0, 1]]]) * christ_yyy(base_point)

        c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8

        return c

    base_point = torch.tensor([1., 1.])

    gamma_hand = hand_christoffels(base_point)
    gamma_derivative_hand = jacobian(hand_christoffels, base_point)

    gamma = rm.connection.christoffels(base_point)
    gamma_derivative = jacobian(rm.connection.christoffels, base_point)

    """
    print(gamma)
    print(gamma_hand)

    print(jacobian(christ_xyx, base_point))
    print(gamma_derivative_hand[0, 1, 0, :])
    print("\n")

    print(jacobian(christ_xyy, base_point))
    print(gamma_derivative_hand[0, 1, 1, :])
    print("\n")

    print(jacobian(christ_xxx, base_point))
    print(gamma_derivative_hand[0, 0, 0, :])
    print("\n")

    print(jacobian(christ_yyy, base_point))
    print(gamma_derivative_hand[1, 1, 1, :])

    print(gamma_derivative)
    print(gamma_derivative_hand)

    print(torch.eq(gamma, gamma_hand))
    print(torch.dist(gamma, gamma_hand) / torch.max(gamma))
    print(torch.dist(gamma_derivative, gamma_derivative_hand) / torch.max(gamma_derivative))
    """

    rm.sectional_curvature(torch.tensor([1., 0.]), torch.tensor([0., 1.]), base_point)
    # curvature = torch.empty(coordinates.shape[0])

    # percent = -1

    # for i, coordinate in enumerate(coordinates):
    #    new_percent = np.ceil(i * 100 / num_coordinates)
    #    if new_percent != percent:
    #        percent = new_percent
    #        print(f"[Curv] {percent}%")
    #    curvature[i] = rm.sectional_curvature(torch.tensor([1., 0.]), torch.tensor([0., 1.]), coordinate)

    # fig, ax = plt.subplots()
    # scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=curvature)

    # legend = ax.legend(*scatter.legend_elements(), title="labels")
    # ax.set_aspect("equal")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # fig.suptitle(f"Sectional Curvature")
    # plt.savefig(
    #    f"/export/home/pnazari/workspace/AutoEncoderVisualization/output/tests/curvature/{encoder_name}/curvature.png")
    # plt.show()


def test_determinants():
    # create output directory if not exists
    Path(f"/stuff/old/tests/determinants/{encoder_name}/").mkdir(
        parents=True,
        exist_ok=True)

    x_min = 1
    x_max = 3
    y_min = 1
    y_max = 3

    grid_steps = 50

    # generate coordinate grid
    grid_x, grid_y = torch.meshgrid(torch.linspace(x_min, x_max, steps=grid_steps),
                                    torch.linspace(y_min, y_max, steps=grid_steps),
                                    indexing="ij")

    # initialize diffgeo objects
    pbm = PullbackMetric(2, decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    # prepare coordinate tuples
    coordinates = torch.vstack([grid_x.ravel(), grid_y.ravel()]).T

    determinants = torch.empty(coordinates.shape[0])

    for i, coordinate in enumerate(coordinates):
        metric = rm.metric.metric_matrix(coordinate)
        determinants[i] = torch.linalg.det(metric)

    fig, ax = plt.subplots()
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=determinants)

    legend = ax.legend(*scatter.legend_elements(), title="labels")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle(f"Curvature")
    plt.savefig(
        f"/export/home/pnazari/workspace/AutoEncoderVisualization/output/tests/determinants/{encoder_name}/det.png")
    plt.show()


def test_island():
    # create output directory if not exists
    Path(f"/stuff/old/tests/error/{encoder_name}/").mkdir(
        parents=True,
        exist_ok=True)

    # initialize diffgeo objects
    pbm = PullbackMetric(2, decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    start = torch.tensor([0.1, 0.1])

    unit_vectors = rm.generate_unit_vectors(10, start)

    dist = []
    error = []

    x_start = 2.

    for i in range(1, 20):
        print(f"{i + 1} of {20}")
        # set bounding box
        x_min = x_start
        x_max = x_start + i * 0.5

        # start and end of geodesic
        start = torch.tensor([x_min, 1])
        end = torch.tensor([x_max, 1])
        direc = end - start
        direc = direc / rm.metric.norm(direc, start) * rm.metric.dist(start, end)

        # points, geodesic, directions = rm.points_along_geodesic(start, stepsize, end_point=end, directions=True)
        geodesic = rm.connection.geodesic(start, initial_tangent_vec=direc)

        dist.append(end[0] - start[0])
        # error.append((rm.metric.dist(start, geodesic(1.))- rm.metric.dist(start, end)))/rm.metric.dist(start, end))

        error.append(abs((geodesic(1.) - end)[0]) / (end[0] - start[0]))

    dist = torch.stack(dist)
    error = torch.stack(error)

    print(error)

    fig = plt.figure()
    plt.plot(dist, error)
    fig.suptitle(f"Geodesic Error, start at x={x_start}")
    plt.xlabel("distance")
    plt.ylabel("relative error in end of geodesic")
    plt.savefig(
        f"/export/home/pnazari/workspace/AutoEncoderVisualization/output/tests/error/{encoder_name}/error_at_{x_start}.png")
    plt.show()


def test_coordinate_grid():
    # TODO: fix this issue with non-invertibility I think. Just catch the error
    # TODO: give measure for quality of transport by error of angle
    # TODO: grid in beide richtungen anpassen
    # TODO: display sectional curvature in color map
    # TODO: first parallel transport along one of the unit vectors! Could also define one of them this way... THIS IS GOOD!
    # TODO: ... one can then choose the orientation of the second one accordingly!
    # TODO: maybe do Isolines instead?
    # TODO: two bugs:
    #    - I should do a parallel transport on the left up and then calculate geodesics following the vector to the right, instead of conencting
    #    - Second vector should point into the first quadrant!

    # create output directory if not exists
    Path(f"/stuff/old/tests/grid/{encoder_name}/ladder").mkdir(
        parents=True,
        exist_ok=True)

    # initialize diffgeo objects
    pbm = PullbackMetric(2, decoder)
    lcc = LeviCivitaConnection(2, pbm)
    rm = RiemannianManifold(2, (1, 1), metric=pbm, connection=lcc)

    n_rungs = 2
    verbose = True

    # number of steps to calculate on one geodesic

    # set bounding box
    x_min = 6.
    x_max = 30.
    y_min = 1.
    y_max = 8.

    # start and end of geodesic
    start = torch.tensor([x_min, y_min])
    end = torch.tensor([x_max, y_min])

    # size of steps on geodesics (geodesic distance)
    stepsize = 20

    starting_frame = rm.metric.get_orthonormal_system(torch.tensor([0., 1.]), start)

    if torch.linalg.det(starting_frame) == 0:
        print(f"[ERROR] Please pick a starting point at which the metric is nonsingular")
        return

    # Do one horizontal parallel transport

    # get points and directions along geodesic
    print(f"{Color.BLUE}[Frame] Lower Horizontal{Color.NC}")

    direc = end - start
    direc = direc / rm.metric.norm(direc, start) * rm.metric.dist(start, end)

    # points, geodesic, directions = rm.points_along_geodesic(start, stepsize, end_point=end, directions=True)
    points, geodesic, directions = rm.points_along_geodesic(start, stepsize, initial_tangent_vec=direc, directions=True)

    rr = rm.time_at_dist(start, rm.metric.dist(start, end), geodesic, 0.)

    t = np.linspace(0, 3, 100)
    r = geodesic(t)
    plt.scatter(r[:, 0], r[:, 1])
    plt.show()

    print(direc)
    print(rr)
    print(geodesic(rr))
    print(end)
    print(geodesic(1.))
    print(rm.metric.dist(start, geodesic(rr)))
    print(rm.metric.dist(start, geodesic(1)))
    print(rm.metric.dist(start, end))
    print(rm.metric.norm(direc, start))
    sys.exit()

    # print(geodesic(1.))

    # parallel transport frame
    frame = rm.parallel_transport_frame(points, directions, starting_frame, n_rungs=n_rungs, verbose=verbose)

    # leftmost and rightmost geodesic
    scaling_factor = rm.metric.dist(torch.tensor([x_min, y_min]), torch.tensor([x_max, y_max]))
    left_initial_tangent_vec = frame[0][1] * scaling_factor
    right_initial_tangent_vec = frame[-1][1] * scaling_factor

    print(f"{Color.BLUE}[Frame] Left Vertical{Color.NC}")
    left_points, left_geodesic = rm.points_along_geodesic(points[0, :],
                                                          stepsize,
                                                          initial_tangent_vec=left_initial_tangent_vec,
                                                          upper_bound=y_max)

    print(f"{Color.BLUE}[Frame] Right Vertical{Color.NC}")
    right_points, right_geodesic = rm.points_along_geodesic(points[-1, :],
                                                            stepsize,
                                                            initial_tangent_vec=right_initial_tangent_vec,
                                                            upper_bound=y_max)

    horizontal_geodesics = []
    vertical_geodesics = []

    # generate the horizontal geodesics
    for row in range(min(left_points.shape[0], right_points.shape[0])):
        print(f"[Grid] Horizontal: {row + 1} of {left_points.shape[0]}")
        geodesic_row = rm.connection.geodesic(left_points[row], end_point=right_points[row])
        horizontal_geodesics.append(geodesic_row)

    # generate vertical geodesics
    for col in range(points.shape[0]):
        print(f"[Grid] Vertical: {col + 1} of {right_points.shape[0]}")
        # TODO: use real scaling factor here! Maybe also calculate the distances for upper frame piece?
        initial_tangent_vec = frame[col][1] * scaling_factor
        geodesic_col = rm.connection.geodesic(points[col, :], initial_tangent_vec=initial_tangent_vec)
        vertical_geodesics.append(geodesic_col)

    print("[Plotting]...")

    # plot vector field
    qv_kwargs = {
        "angles": 'xy',
        "scale_units": 'xy',
        "scale": 3,
        "headwidth": 2,
        "headlength": 2,
        "width": 0.005,
        "color": "navy",
    }

    frame = frame.detach()

    fig = plt.figure()

    time = np.linspace(0, 1, 20)

    # plot rows:
    for geodesic_row in horizontal_geodesics:
        points_row = geodesic_row(time).detach()
        plt.plot(points_row[:, 0], points_row[:, 1], c="navy")

    # plot columns
    for geodesic_col in vertical_geodesics:
        points_col = geodesic_col(time).detach()
        plt.plot(points_col[:, 0], points_col[:, 1], c="navy")

    # plot frame in lowest row
    plt.quiver(points[:, 0], points[:, 1], frame[:, 0, :][:, 0], frame[:, 0, :][:, 1], **qv_kwargs)
    plt.quiver(points[:, 0], points[:, 1], frame[:, 1, :][:, 0], frame[:, 1, :][:, 1], **qv_kwargs)

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    fig.suptitle(f"Parallel Frames")
    plt.savefig(
        f"/export/home/pnazari/workspace/AutoEncoderVisualization/output/tests/grid/{encoder_name}/ladder/{encoder_name}_parallel.png")
    plt.show()


umap_zilionis()
# tsne_umap()
