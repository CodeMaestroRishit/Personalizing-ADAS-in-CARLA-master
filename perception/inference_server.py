"""TCP inference service for the legacy CARLA client process."""

import argparse
import json
import socketserver
import struct
import time

import numpy as np
import torch
from PIL import Image

from perception.models import ConvolutionalBaseline, SpikingConvNet


def receive_exact(connection, byte_count):
    chunks = []
    received_count = 0
    while received_count < byte_count:
        chunk = connection.recv(byte_count - received_count)
        if not chunk:
            raise IOError('Client disconnected during request')
        chunks.append(chunk)
        received_count += len(chunk)
    return b''.join(chunks)


class InferenceHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                request_length = struct.unpack('!I', receive_exact(self.request, 4))[0]
                request_payload = receive_exact(self.request, request_length)
            except (IOError, OSError, struct.error):
                return
            if len(request_payload) != 128 * 128 + 1:
                response_body = json.dumps({'error': 'Expected one 128x128 grayscale frame'}).encode('utf-8')
                self.request.sendall(struct.pack('!I', len(response_body)) + response_body)
                continue
            model_identifier = request_payload[0]
            requested_model_name = 'snn' if model_identifier == 1 else 'cnn' if model_identifier == 2 else None
            if requested_model_name != self.server.model_name:
                response_body = json.dumps({
                    'error': 'Client requested {} but service runs {}'.format(
                        requested_model_name, self.server.model_name
                    )
                }).encode('utf-8')
                self.request.sendall(struct.pack('!I', len(response_body)) + response_body)
                continue
            frame_image = Image.frombytes('L', (128, 128), request_payload[1:])
            frame_array = np.asarray(frame_image, dtype=np.float32) / 255.0
            batch_of_input_frames = torch.from_numpy(frame_array).unsqueeze(0).unsqueeze(0)
            batch_of_input_frames = batch_of_input_frames.to(self.server.device)
            if self.server.device.type == 'cuda':
                torch.cuda.synchronize(self.server.device)
            inference_started_at = time.perf_counter()
            with torch.no_grad():
                if self.server.model_name == 'snn':
                    logits = self.server.model(
                        batch_of_input_frames,
                        num_time_steps=self.server.num_time_steps,
                    )
                else:
                    logits = self.server.model(batch_of_input_frames)
                obstacle_probability = torch.softmax(logits, dim=1)[0, 1].item()
            if self.server.device.type == 'cuda':
                torch.cuda.synchronize(self.server.device)
            inference_latency_ms = (time.perf_counter() - inference_started_at) * 1000.0
            response_body = json.dumps({
                'obstacle_probability': obstacle_probability,
                'inference_latency_ms': inference_latency_ms,
            }).encode('utf-8')
            self.request.sendall(struct.pack('!I', len(response_body)) + response_body)


class InferenceServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def parse_arguments():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument('--model', choices=('snn', 'cnn'), required=True)
    argument_parser.add_argument('--checkpoint', default=None)
    argument_parser.add_argument('--host', default='127.0.0.1')
    argument_parser.add_argument('--port', type=int, default=8765)
    argument_parser.add_argument('--device', default='auto')
    argument_parser.add_argument('--seed', type=int, default=1337)
    return argument_parser.parse_args()


def main():
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    checkpoint_path = arguments.checkpoint or 'perception/checkpoints/{}.pt'.format(arguments.model)
    device = torch.device(
        'cuda' if arguments.device == 'auto' and torch.cuda.is_available()
        else 'cpu' if arguments.device == 'auto' else arguments.device
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if arguments.model == 'snn':
        model = SpikingConvNet(
            beta=checkpoint.get('beta', 0.9),
            threshold=checkpoint.get('threshold', 1.0),
            num_time_steps=checkpoint.get('num_time_steps', 8),
            encoding=checkpoint.get('encoding', 'rate'),
        )
    else:
        model = ConvolutionalBaseline()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    service = InferenceServer((arguments.host, arguments.port), InferenceHandler)
    service.model_name = arguments.model
    service.model = model
    service.device = device
    service.num_time_steps = checkpoint.get('num_time_steps', 8)
    print('Serving {} inference on {}:{} using {}'.format(
        arguments.model, arguments.host, arguments.port, device
    ))
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.server_close()


if __name__ == '__main__':
    main()
