"""Temporal spike encoders for grayscale image tensors."""

import torch


def rate_encode(batch_of_input_frames, num_time_steps, generator=None):
    """Return Bernoulli/Poisson rate spikes shaped (time, batch, channels, H, W).

    Pixel intensities must be floating point values in [0, 1]. A Bernoulli draw
    per timestep is the discrete-time Poisson approximation commonly used for
    rate coding.
    """
    if num_time_steps < 1:
        raise ValueError("num_time_steps must be positive")
    frame_rates = batch_of_input_frames.clamp(0.0, 1.0)
    random_values = torch.rand(
        (num_time_steps,) + tuple(frame_rates.shape),
        dtype=frame_rates.dtype,
        device=frame_rates.device,
        generator=generator,
    )
    return (random_values < frame_rates.unsqueeze(0)).to(frame_rates.dtype)


def latency_encode(batch_of_input_frames, num_time_steps):
    """Encode intensities as a single time-to-first-spike event.

    Brighter pixels spike earlier. Zero-valued pixels do not spike; nonzero
    pixels emit one event at floor((1 - intensity) * num_time_steps).
    """
    if num_time_steps < 1:
        raise ValueError("num_time_steps must be positive")
    frame_rates = batch_of_input_frames.clamp(0.0, 1.0)
    first_spike_time = torch.floor((1.0 - frame_rates) * num_time_steps).long()
    first_spike_time = first_spike_time.clamp(max=num_time_steps - 1)
    time_indices = torch.arange(
        num_time_steps, device=frame_rates.device
    ).view((num_time_steps,) + (1,) * frame_rates.dim())
    spike_recording = (time_indices == first_spike_time.unsqueeze(0))
    return (spike_recording & (frame_rates.unsqueeze(0) > 0)).to(frame_rates.dtype)


def encode_frames(batch_of_input_frames, num_time_steps, encoding="rate", generator=None):
    """Dispatch to rate or latency encoding."""
    if encoding == "rate":
        return rate_encode(batch_of_input_frames, num_time_steps, generator)
    if encoding == "latency":
        return latency_encode(batch_of_input_frames, num_time_steps)
    raise ValueError("encoding must be 'rate' or 'latency'")
