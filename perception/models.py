"""SNN and matched CNN classifiers for 128x128 grayscale camera frames."""

import torch
from torch import nn

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError:  # permit the CNN and data tools without snnTorch installed
    snn = None
    surrogate = None

from perception.encoding import encode_frames


def _require_snntorch():
    if snn is None:
        raise ImportError("SpikingConvNet requires snnTorch; install requirements-perception.txt")


class SpikingConvNet(nn.Module):
    """Conv-LIF -> Conv-LIF -> Linear-LIF network with temporal readout."""

    def __init__(self, beta=0.9, threshold=1.0, num_time_steps=8, encoding="rate"):
        super(SpikingConvNet, self).__init__()
        _require_snntorch()
        self.num_time_steps = int(num_time_steps)
        self.encoding = encoding
        self.convolution_1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.convolution_2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.classifier = nn.Linear(32 * 32 * 32, 2)
        spike_gradient = surrogate.fast_sigmoid()
        self.lif_1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_gradient)
        self.lif_2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_gradient)
        self.lif_output = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_gradient)

    def forward(self, batch_of_input_frames, num_time_steps=None,
                return_spike_recording=False):
        simulation_steps = self.num_time_steps if num_time_steps is None else int(num_time_steps)
        if simulation_steps < 1:
            raise ValueError("num_time_steps must be positive")
        encoded_frames = encode_frames(
            batch_of_input_frames, simulation_steps, encoding=self.encoding
        )
        membrane_potential_1 = None
        membrane_potential_2 = None
        membrane_potential_output = None
        output_spike_count = batch_of_input_frames.new_zeros((batch_of_input_frames.size(0), 2))
        spike_count_per_layer = {
            "conv1_lif": batch_of_input_frames.new_zeros(()),
            "conv2_lif": batch_of_input_frames.new_zeros(()),
            "linear_lif": batch_of_input_frames.new_zeros(()),
        }

        for time_index in range(simulation_steps):
            convolution_1_output = self.convolution_1(encoded_frames[time_index])
            spike_recording_per_layer_1, membrane_potential_1 = self.lif_1(
                convolution_1_output, membrane_potential_1
            )
            convolution_2_output = self.convolution_2(spike_recording_per_layer_1)
            spike_recording_per_layer_2, membrane_potential_2 = self.lif_2(
                convolution_2_output, membrane_potential_2
            )
            flattened_spikes = torch.flatten(spike_recording_per_layer_2, start_dim=1)
            linear_output = self.classifier(flattened_spikes)
            output_spikes, membrane_potential_output = self.lif_output(
                linear_output, membrane_potential_output
            )
            output_spike_count = output_spike_count + output_spikes
            spike_count_per_layer["conv1_lif"] += spike_recording_per_layer_1.detach().sum()
            spike_count_per_layer["conv2_lif"] += spike_recording_per_layer_2.detach().sum()
            spike_count_per_layer["linear_lif"] += output_spikes.detach().sum()

        logits = output_spike_count / float(simulation_steps)
        if return_spike_recording:
            return logits, spike_count_per_layer
        return logits


class ConvolutionalBaseline(nn.Module):
    """CNN baseline with the same convolution and classifier dimensions."""

    def __init__(self):
        super(ConvolutionalBaseline, self).__init__()
        self.convolution_1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.activation_1 = nn.ReLU(inplace=False)
        self.convolution_2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.activation_2 = nn.ReLU(inplace=False)
        self.classifier = nn.Linear(32 * 32 * 32, 2)

    def forward(self, batch_of_input_frames):
        convolution_1_output = self.activation_1(self.convolution_1(batch_of_input_frames))
        convolution_2_output = self.activation_2(self.convolution_2(convolution_1_output))
        return self.classifier(torch.flatten(convolution_2_output, start_dim=1))
