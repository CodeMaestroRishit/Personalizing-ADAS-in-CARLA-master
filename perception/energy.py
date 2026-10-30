"""Operation counters used for transparent energy proxy estimates."""

import torch
import torch.nn as nn


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def count_cnn_macs(model, input_shape=(1, 1, 128, 128)):
    """Count multiply-accumulates for one forward pass using module shapes."""
    output_macs = []
    hooks = []

    def count_convolution(module, inputs, output):
        output_positions = output.numel() // output.shape[1]
        macs = output_positions * module.out_channels * (
            module.in_channels // module.groups
        ) * module.kernel_size[0] * module.kernel_size[1]
        output_macs.append(int(macs))

    def count_linear(module, inputs, output):
        output_macs.append(int(module.in_features * module.out_features))

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(count_convolution))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(count_linear))
    try:
        parameter = next(model.parameters())
        device = parameter.device
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(torch.zeros(input_shape, device=device))
        model.train(was_training)
    finally:
        for hook in hooks:
            hook.remove()
    return sum(output_macs)


def count_synaptic_operations(spike_count_per_layer, fan_out_per_layer,
                              batch_size=1, num_time_steps=1):
    """Return recorded output spikes times downstream fan-out per input frame.

    Spike counts include all simulation steps, so only the batch dimension is
    normalized; time is part of the work performed for one inference.
    """
    synaptic_operations = 0.0
    for layer_name, spike_count in spike_count_per_layer.items():
        fan_out = fan_out_per_layer[layer_name]
        if hasattr(spike_count, "item"):
            spike_count = spike_count.item()
        synaptic_operations += float(spike_count) * float(fan_out)
    normalizer = max(1, int(batch_size))
    return synaptic_operations / normalizer


def snn_fan_out(model):
    """Approximate full-kernel fan-out from LIF outputs to the next layer."""
    return {
        "conv1_lif": model.convolution_2.out_channels
        * model.convolution_2.kernel_size[0] * model.convolution_2.kernel_size[1],
        "conv2_lif": model.classifier.out_features,
        # Output events are the readout; no downstream synapses are counted.
        "linear_lif": 0,
    }
