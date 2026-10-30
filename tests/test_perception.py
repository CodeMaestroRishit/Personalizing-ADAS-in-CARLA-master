"""Unit tests for perception encoding, model shapes, and operation estimates."""

import unittest

import torch

from perception.encoding import latency_encode, rate_encode
from perception.energy import count_synaptic_operations
from perception.models import ConvolutionalBaseline, SpikingConvNet


class EncodingTests(unittest.TestCase):
    def test_rate_encoder_shape_and_binary_values(self):
        input_frames = torch.full((2, 1, 4, 4), 0.5)
        encoded_frames = rate_encode(input_frames, 5, generator=torch.Generator().manual_seed(7))
        self.assertEqual(tuple(encoded_frames.shape), (5, 2, 1, 4, 4))
        self.assertTrue(torch.all((encoded_frames == 0) | (encoded_frames == 1)))

    def test_latency_encoder_spikes_once_and_brighter_is_earlier(self):
        input_frames = torch.tensor([[[[0.0, 0.5, 1.0]]]])
        encoded_frames = latency_encode(input_frames, 4)
        self.assertEqual(int(encoded_frames[:, 0, 0, 0, 0].sum()), 0)
        self.assertEqual(int(encoded_frames[:, 0, 0, 0, 1].sum()), 1)
        self.assertEqual(int(encoded_frames[:, 0, 0, 0, 2].sum()), 1)
        self.assertEqual(int(encoded_frames[:, 0, 0, 0, 1].argmax()), 2)
        self.assertEqual(int(encoded_frames[:, 0, 0, 0, 2].argmax()), 0)


class ModelShapeTests(unittest.TestCase):
    def test_cnn_forward_shape(self):
        model = ConvolutionalBaseline()
        predictions = model(torch.zeros(2, 1, 128, 128))
        self.assertEqual(tuple(predictions.shape), (2, 2))

    def test_snn_forward_shape_and_spike_recording(self):
        model = SpikingConvNet(num_time_steps=2, encoding="latency")
        predictions, spike_recording_per_layer = model(
            torch.zeros(2, 1, 128, 128), return_spike_recording=True
        )
        self.assertEqual(tuple(predictions.shape), (2, 2))
        self.assertEqual(set(spike_recording_per_layer), {"conv1_lif", "conv2_lif", "linear_lif"})


class SynapticOperationTests(unittest.TestCase):
    def test_synops_are_spikes_times_fan_out_per_example(self):
        synaptic_operations = count_synaptic_operations(
            {"conv1_lif": 100, "conv2_lif": 20},
            {"conv1_lif": 4, "conv2_lif": 2},
            batch_size=2,
        )
        self.assertEqual(synaptic_operations, 220)


if __name__ == "__main__":
    unittest.main()
