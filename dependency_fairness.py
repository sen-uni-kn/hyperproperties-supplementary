# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from adult import Adult


dataset = Adult(".datasets")
sensitive_i1, sensitive_i2 = dataset.sensitive_column_indices


class NetIn1(nn.Module):
    """
    First variant of the input network.
    Uses element assignment.
    """

    def __call__(self, w):
        """
        Transforms an auxiliary input into two actual inputs
        that are identical, except for the (binary) sensitive attribute.
        The values of the sensitive attribute are complementary.

        :param w: Auxiliary input :math:`w`
        :return: Actual network inputs :math:`x_1, x_2`
        """
        x1 = w
        x2 = torch.clone(w)

        # set sensitive attribute (one-hot encoded binary attribute)
        # assume batched input
        x1[:, sensitive_i1] = 1.0
        x1[:, sensitive_i2] = 0.0

        x2[:, sensitive_i1] = 0.0
        x2[:, sensitive_i2] = 1.0

        return x1, x2


class NetIn2(nn.Module):
    """
    Second variant of the input network.
    Uses an affine transformation instead of element assignment.
    """

    def __call__(self, w):
        """
        Transforms an auxiliary input into two actual inputs
        that are identical, except for the (binary) sensitive attribute.
        The values of the sensitive attribute are complementary.

        :param w: Auxiliary input :math:`w`
        :return: Actual network inputs :math:`x_1, x_2`
        """
        x1 = w
        x2 = torch.clone(w)

        # set sensitive attribute (one-hot encoded binary attribute)
        # using affine transformations
        A = torch.eye(w.shape[-1], dtype=w.dtype, device=w.device)
        A[sensitive_i1, sensitive_i1] = 0.0
        A[sensitive_i2, sensitive_i2] = 0.0
        b1 = torch.zeros(w.shape[-1], dtype=w.dtype, device=w.device)
        b1[sensitive_i1] = 1.0
        b2 = torch.zeros(w.shape[-1], dtype=w.dtype, device=w.device)
        b2[sensitive_i2] = 1.0

        x1 = x1 @ A + b1  # x1 may be batched
        x2 = x2 @ A + b2
        return x1, x2


class NetSat1(nn.Module):
    """
    First variant of the satisfaction network.
    Uses element access, max, and min.
    """

    def __call__(self, y1, y2):
        """
        Computes a satisfaction function for the dependency fairness
        output set :math:`\mathcal{Y} = \{ y_1, y_1 \mid arg\_max(y_1) = arg\_max(y_2) \}`.
        Concretely, returns a non-negative value only if :math:`y1` and :math:`y2`
        share a maximal element (same classification).


        :param y1: First network output :math:`y_1`
        :param y2: Second network output :math:`y_2`
        :return: Returns a non-negative value if :math:`y1` and :math:`y2`
        share a maximal element and a negative value otherwise.
        """
        num_outputs = y1.shape[-1]
        # the expression arg_max(y1) == arg_max(y2) is equivalent to:
        # OR_{i in 1 ... num_outputs)
        #       AND_{j in 1 ... num_outputs, j!=i} y1_i > y1_j
        #   AND
        #       AND_{j in 1 ... num_outputs, j!=i} y2_i > y2_j
        # we use the usual OR => max, AND => min construction
        # and take a shortcut for the inner AND
        values = []
        for i in range(num_outputs):
            select_others = torch.arange(num_outputs) != i
            target_output_1 = y1[:, i]
            target_output_2 = y2[:, i]
            other_outputs_1 = y1[:, select_others]
            other_outputs_2 = y2[:, select_others]
            value_1 = target_output_1 - torch.max(other_outputs_1, dim=1).values
            value_2 = target_output_2 - torch.max(other_outputs_2, dim=1).values
            value = torch.min(value_1, value_2)
            values.append(value)
        return torch.max(torch.stack(values, dim=-1), dim=-1).values


class NetSat2(nn.Module):
    """
    Second variant of the satisfaction network.
    Uses affine computations, max pooling,
    reshaping, and stacking (:code:`torch.stack`).
    """

    def __call__(self, y1, y2):
        """
        Computes a satisfaction function for the dependency fairness
        output set :math:`\mathcal{Y} = \{ y_1, y_1 \mid arg\_max(y_1) = arg\_max(y_2) \}`.
        Concretely, returns a non-negative value only if :math:`y1` and :math:`y2`
        share a maximal element (same classification).

        :param y1: First network output :math:`y_1`
        :param y2: Second network output :math:`y_2`
        :return: Returns a non-negative value if :math:`y1` and :math:`y2`
        share a maximal element and a negative value otherwise.
        """
        num_outputs = int(y1.shape[-1])

        # create a 3d tensor with a channel dimension that contains y1/y2
        ys = torch.stack((y1, y2), dim=1)
        values = []
        for i in range(num_outputs):
            # reduce y1 and y2 to y1[:, i] and y2[:, i]
            A1 = torch.zeros((num_outputs, 1), dtype=y1.dtype, device=y1.device)
            A1[i] = 1.0
            # replace entry i of y1 and y2 with -inf (for max)
            A2 = torch.eye(num_outputs, dtype=y1.dtype, device=y1.device)
            A2[i, i] = 0.0
            b2 = torch.zeros(num_outputs, dtype=y1.dtype, device=y1.device)
            b2[i] = -torch.inf

            # add channel dimension for max pooling
            maxs = F.max_pool1d(ys @ A2 + b2, num_outputs)  # shape: N (batch), 2, 1
            value = ys @ A1 - maxs  # shape: N, 2, 1

            # min over y1 and y2
            value = value.reshape(-1, 1, 2)
            # min(a, b) = -max(-a, -b)
            value = -F.max_pool1d(-value, 2)  # shape: N, 1, 1
            values.append(value)
        values = torch.stack(values, -1)  # shape: N, 1, 1, num_outputs
        values = values.reshape(-1, 1, num_outputs)
        f_sat = F.max_pool1d(values, num_outputs)  # shape: N, 1, 1
        return f_sat.reshape(-1)


if __name__ == "__main__":
    torch.manual_seed(246931270537065)
    parser = ArgumentParser(
        "Dependency Fairness Verification",
        description="Create and export computational graphs for verifying "
        "the dependency fairness NNDH. This script creates two variants"
        "of the dependency fairness NNDH verification computational graph. "
        "The first variant uses more non-standard operations that allow "
        "to write the input and satisfaction networks more naturally. "
        "The second variant uses a restricted set of operations that "
        "are more common for neural networks. The script exports both "
        "computational graphs as ONNX files.",
    )
    parser.add_argument(
        "--network", type=str, help="The path to the network to verify."
    )
    parser.add_argument(
        "--no_falsify", action="store_true", help="Skip falisifying Dependency Fairness"
    )
    args = parser.parse_args()

    network = torch.load(args.network)

    class NNDHNetwork(nn.Module):
        def __init__(self, net, net_in, net_sat):
            super().__init__()
            self.net = net
            self.net_in = net_in
            self.net_sat = net_sat

        def __call__(self, w):
            x1, x2 = self.net_in(w)
            y1 = self.net(x1)
            y2 = self.net(x2)
            f_sat = self.net_sat(y1, y2)
            return f_sat

    nndh_net_1 = NNDHNetwork(network, NetIn1(), NetSat1())
    nndh_net_2 = NNDHNetwork(network, NetIn2(), NetSat2())

    W = 100 * torch.rand((10, dataset.data.size(-1))) - 50
    assert torch.allclose(nndh_net_1(W), nndh_net_2(W), atol=1e-6)
    assert torch.allclose(nndh_net_1(dataset.data), nndh_net_2(dataset.data), atol=1e-6)

    # falsify using PGD
    mins = dataset.data.min(dim=0).values
    maxs = dataset.data.max(dim=0).values

    if not args.no_falsify:
        print("Falsifying Dependency Fairness using PGD.")
        for _ in tqdm(range(100)):  # 100 restarts
            w = mins + (maxs - mins) * torch.rand_like(mins)
            w = w.unsqueeze(0)
            optim = torch.optim.SGD((w,), lr=(maxs - mins).mean() / 100)
            for _ in range(100):  # 100 iterations
                optim.zero_grad()
                f_sat = nndh_net_1(w)
                f_sat2 = nndh_net_2(w)
                f_sat.backward()  # minimise f_sat => maximise violation
                optim.step()
                # project into input domain
                w = torch.clip(w, mins, maxs)
            if nndh_net_1(w) < 0.0:
                print(f"Found counterexample: {w}")
                break
        else:
            print("No violation discovered.")

    network_path = Path(args.network)
    network_name = network_path.stem
    out_1_path = Path(network_path.parent, network_name + "_NNDH_1.onnx")
    torch.onnx.export(nndh_net_1, mins.unsqueeze(0), out_1_path)
    out_2_path = Path(network_path.parent, network_name + "_NNDH_2.onnx")
    torch.onnx.export(nndh_net_2, mins.unsqueeze(0), out_2_path)
