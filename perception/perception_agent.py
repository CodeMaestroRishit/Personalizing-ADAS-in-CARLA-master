"""Small CARLA-side client for a separate SNN/CNN inference process.

This module intentionally avoids importing PyTorch so it can run alongside the
CARLA 0.9.8 Python client egg (commonly built for Python 3.5).
"""

import json
import socket
import struct
import time

import numpy as np
from PIL import Image


class PerceptionAgent(object):
    def __init__(self, model='snn', host='127.0.0.1', port=8765, timeout=1.0):
        if model not in ('snn', 'cnn'):
            raise ValueError("model must be 'snn' or 'cnn'")
        self.model = model
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.connection = None
        self.last_latency_ms = None
        self.last_round_trip_ms = None

    def _connect(self):
        if self.connection is None:
            self.connection = socket.create_connection((self.host, self.port), self.timeout)
            self.connection.settimeout(self.timeout)

    @staticmethod
    def _receive_exact(connection, byte_count):
        chunks = []
        received_count = 0
        while received_count < byte_count:
            chunk = connection.recv(byte_count - received_count)
            if not chunk:
                raise IOError('Inference service closed the connection')
            chunks.append(chunk)
            received_count += len(chunk)
        return b''.join(chunks)

    @staticmethod
    def _to_grayscale(camera_frame):
        if isinstance(camera_frame, np.ndarray):
            frame_array = camera_frame
            if frame_array.ndim == 2:
                grayscale_image = Image.fromarray(frame_array.astype(np.uint8), mode='L')
            else:
                grayscale_image = Image.fromarray(frame_array.astype(np.uint8), mode='RGB').convert('L')
        elif hasattr(camera_frame, 'raw_data'):
            frame_array = np.frombuffer(camera_frame.raw_data, dtype=np.uint8)
            frame_array = frame_array.reshape((camera_frame.height, camera_frame.width, 4))
            rgb_array = frame_array[:, :, :3][:, :, ::-1].copy()
            grayscale_image = Image.fromarray(rgb_array, mode='RGB').convert('L')
        else:
            grayscale_image = camera_frame.convert('L')
        grayscale_image = grayscale_image.resize((128, 128), Image.BILINEAR)
        return np.asarray(grayscale_image, dtype=np.uint8).tobytes()

    def predict(self, camera_frame):
        """Return class-1 obstacle probability for a CARLA image or RGB array."""
        grayscale_bytes = self._to_grayscale(camera_frame)
        self._connect()
        model_identifier = 1 if self.model == 'snn' else 2
        request_payload = struct.pack('!B', model_identifier) + grayscale_bytes
        request = struct.pack('!I', len(request_payload)) + request_payload
        started_at = time.time()
        try:
            self.connection.sendall(request)
            response_length = struct.unpack('!I', self._receive_exact(self.connection, 4))[0]
            response = json.loads(self._receive_exact(self.connection, response_length).decode('utf-8'))
        except (IOError, OSError, ValueError, socket.timeout):
            self.close()
            raise
        if 'error' in response:
            self.close()
            raise IOError(response['error'])
        self.last_round_trip_ms = (time.time() - started_at) * 1000.0
        self.last_latency_ms = float(response['inference_latency_ms'])
        return float(response['obstacle_probability'])

    def close(self):
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None


if __name__ == '__main__':
    raise SystemExit('Use PerceptionAgent from environment/ego_vehicle.py; start perception.inference_server separately.')
