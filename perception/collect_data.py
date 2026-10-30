#!/usr/bin/env python
"""Collect balanced grayscale perception frames and CARLA ground-truth labels.

Run this script with the Python interpreter compatible with the CARLA 0.9.8
client egg. The collector makes the world synchronous at 10 Hz so image frames
and actor transforms used for labels correspond to the same simulation tick.
"""

import argparse
import csv
import glob
import math
import os
import random
import sys
import time
try:
    from Queue import Empty, Queue
except ImportError:
    from queue import Empty, Queue

try:
    import numpy as np
except ImportError:
    np = None

try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carla_egg_directory = os.path.join(
        project_root, '..', 'CARLA_Simulator', 'PythonAPI', 'carla', 'dist'
    )
    sys.path.append(glob.glob('../../CARLA_Simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    try:
        sys.path.append(glob.glob(os.path.join(carla_egg_directory, 'carla-*%d.%d-%s.egg' % (
            sys.version_info.major, sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
    except IndexError:
        pass

import carla
from PIL import Image


def parse_arguments():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument('--host', default='127.0.0.1')
    argument_parser.add_argument('--port', type=int, default=2000)
    argument_parser.add_argument('--traffic-manager-port', type=int, default=8000)
    argument_parser.add_argument('--output-root', default='data/perception')
    argument_parser.add_argument('--split', choices=('train', 'val', 'test', 'all'), default='all')
    argument_parser.add_argument('--samples-per-class', type=int, default=500)
    argument_parser.add_argument('--max-frames-per-split', type=int, default=20000)
    argument_parser.add_argument('--seed', type=int, default=1337)
    argument_parser.add_argument('--timeout', type=float, default=10.0)
    return argument_parser.parse_args()


def is_obstacle_vehicle_ahead(carla_world, ego_vehicle, distance_limit=25.0):
    ego_transform = ego_vehicle.get_transform()
    ego_waypoint = carla_world.get_map().get_waypoint(ego_transform.location)
    forward_vector = ego_transform.get_forward_vector()
    for actor in carla_world.get_actors().filter('vehicle.*'):
        if actor.id == ego_vehicle.id:
            continue
        actor_location = actor.get_location()
        actor_waypoint = carla_world.get_map().get_waypoint(actor_location)
        if actor_waypoint is None or ego_waypoint is None:
            continue
        if actor_waypoint.road_id != ego_waypoint.road_id or actor_waypoint.lane_id != ego_waypoint.lane_id:
            continue
        offset_x = actor_location.x - ego_transform.location.x
        offset_y = actor_location.y - ego_transform.location.y
        offset_z = actor_location.z - ego_transform.location.z
        euclidean_distance = math.sqrt(offset_x * offset_x + offset_y * offset_y + offset_z * offset_z)
        forward_distance = (offset_x * forward_vector.x
                            + offset_y * forward_vector.y
                            + offset_z * forward_vector.z)
        if 0.0 < forward_distance <= distance_limit and euclidean_distance <= distance_limit:
            return True
    return False


def choose_spawn_transform(spawn_points, occupied_spawn_indices, random_generator):
    available_indices = [index for index in range(len(spawn_points))
                         if index not in occupied_spawn_indices]
    if not available_indices:
        available_indices = list(range(len(spawn_points)))
    selected_index = random_generator.choice(available_indices)
    return selected_index, spawn_points[selected_index]


def choose_weather(random_generator):
    weather_names = [name for name in dir(carla.WeatherParameters)
                     if name and name[0].isupper() and not name.startswith('_')]
    weather_parameters = [getattr(carla.WeatherParameters, name) for name in weather_names]
    if not weather_parameters:
        return 'ClearNoon', carla.WeatherParameters.ClearNoon
    selected_name = random_generator.choice(weather_names)
    return selected_name, getattr(carla.WeatherParameters, selected_name)


def spawn_background_traffic(carla_world, traffic_manager, ego_vehicle, spawn_points,
                             scene_number, random_generator, traffic_manager_port):
    desired_traffic_counts = {0: (0, 5), 1: (3, 9), 2: (8, 18)}
    minimum_vehicles, maximum_vehicles = desired_traffic_counts[scene_number]
    desired_vehicle_count = random_generator.randint(minimum_vehicles, maximum_vehicles)
    blueprint_library = carla_world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    random_generator.shuffle(spawn_points)
    spawned_vehicles = []
    for spawn_transform in spawn_points:
        if len(spawned_vehicles) >= desired_vehicle_count:
            break
        vehicle_blueprint = random_generator.choice(vehicle_blueprints)
        try:
            background_vehicle = carla_world.try_spawn_actor(vehicle_blueprint, spawn_transform)
        except RuntimeError:
            background_vehicle = None
        if background_vehicle is None or background_vehicle.id == ego_vehicle.id:
            continue
        background_vehicle.set_autopilot(True, traffic_manager_port)
        spawned_vehicles.append(background_vehicle)

    # Scenes 1 and 2 include a deliberately placed same-lane lead vehicle when
    # CARLA accepts the spawn point. Background traffic still supplies variation.
    if scene_number in (1, 2):
        ego_waypoint = carla_world.get_map().get_waypoint(ego_vehicle.get_location())
        forward_waypoints = ego_waypoint.next(15.0) if ego_waypoint is not None else []
        if forward_waypoints:
            lead_blueprint = random_generator.choice(vehicle_blueprints)
            lead_vehicle = carla_world.try_spawn_actor(lead_blueprint, forward_waypoints[0].transform)
            if lead_vehicle is not None:
                lead_vehicle.set_autopilot(True, traffic_manager_port)
                spawned_vehicles.append(lead_vehicle)
    return spawned_vehicles


def connect_camera(carla_world, ego_vehicle, image_queue):
    camera_blueprint = carla_world.get_blueprint_library().find('sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x', '128')
    camera_blueprint.set_attribute('image_size_y', '128')
    camera_blueprint.set_attribute('fov', '90')
    camera_blueprint.set_attribute('sensor_tick', '0.1')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera_sensor = carla_world.spawn_actor(camera_blueprint, camera_transform, attach_to=ego_vehicle)
    camera_sensor.listen(image_queue.put)
    return camera_sensor


def collect_split(client, split_name, arguments, split_seed):
    random_generator = random.Random(split_seed)
    random.seed(split_seed)
    output_root = arguments.output_root
    if not os.path.isabs(output_root):
        output_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_root)
    output_directory = os.path.join(output_root, split_name)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    labels_path = os.path.join(output_directory, 'labels.csv')
    class_counts = {0: 0, 1: 0}
    next_image_index = 0
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as existing_labels_file:
            for existing_row in csv.DictReader(existing_labels_file):
                class_label = int(existing_row['label'])
                if class_label in class_counts:
                    class_counts[class_label] += 1
                    next_image_index += 1
    csv_mode = 'a' if os.path.exists(labels_path) else 'w'
    with open(labels_path, csv_mode) as labels_file:
        label_writer = csv.writer(labels_file)
        if csv_mode == 'w':
            label_writer.writerow(('image', 'label', 'scene', 'weather', 'spawn_index', 'carla_frame'))
        labels_file.flush()

        frames_seen = 0
        episode_index = 0
        while (frames_seen < arguments.max_frames_per_split
               and episode_index < arguments.max_frames_per_split):
            if (class_counts[0] >= arguments.samples_per_class
                    and class_counts[1] >= arguments.samples_per_class):
                break
            carla_world = client.get_world()
            original_settings = carla_world.get_settings()
            synchronous_settings = carla_world.get_settings()
            synchronous_settings.no_rendering_mode = False
            synchronous_settings.synchronous_mode = True
            synchronous_settings.fixed_delta_seconds = 0.1
            carla_world.apply_settings(synchronous_settings)
            traffic_manager = client.get_trafficmanager(arguments.traffic_manager_port)
            try:
                traffic_manager.set_synchronous_mode(True)
                traffic_manager.set_random_device_seed(split_seed + episode_index)
            except (AttributeError, RuntimeError):
                pass

            scene_number = episode_index % 3
            spawn_points = carla_world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError('The active CARLA map has no vehicle spawn points')
            occupied_spawn_indices = set()
            spawn_index, ego_transform = choose_spawn_transform(
                spawn_points, occupied_spawn_indices, random_generator
            )
            occupied_spawn_indices.add(spawn_index)
            ego_blueprint = random_generator.choice(
                carla_world.get_blueprint_library().filter('vehicle.tesla.model3')
            )
            ego_blueprint.set_attribute('role_name', 'ego')
            ego_vehicle = carla_world.try_spawn_actor(ego_blueprint, ego_transform)
            if ego_vehicle is None:
                try:
                    traffic_manager.set_synchronous_mode(False)
                except (AttributeError, RuntimeError):
                    pass
                carla_world.apply_settings(original_settings)
                episode_index += 1
                continue

            camera_sensor = None
            background_vehicles = []
            image_queue = Queue()
            try:
                weather_name, weather_parameters = choose_weather(random_generator)
                carla_world.set_weather(weather_parameters)
                ego_vehicle.set_autopilot(True, arguments.traffic_manager_port)
                camera_sensor = connect_camera(carla_world, ego_vehicle, image_queue)
                background_vehicles = spawn_background_traffic(
                    carla_world, traffic_manager, ego_vehicle, list(spawn_points),
                    scene_number, random_generator, arguments.traffic_manager_port
                )
                episode_frame_count = 0
                episode_frame_limit = 100
                while episode_frame_count < episode_frame_limit:
                    if class_counts[0] >= arguments.samples_per_class and class_counts[1] >= arguments.samples_per_class:
                        break
                    if next_image_index >= arguments.max_frames_per_split:
                        break
                    tick_frame = carla_world.tick(arguments.timeout)
                    try:
                        camera_image = image_queue.get(timeout=arguments.timeout)
                    except Empty:
                        continue
                    while camera_image.frame < tick_frame:
                        try:
                            camera_image = image_queue.get(timeout=arguments.timeout)
                        except Empty:
                            camera_image = None
                            break
                    if camera_image is None or camera_image.frame != tick_frame:
                        continue
                    frames_seen += 1
                    episode_frame_count += 1
                    obstacle_present = is_obstacle_vehicle_ahead(carla_world, ego_vehicle)
                    class_label = 1 if obstacle_present else 0
                    if class_counts[class_label] >= arguments.samples_per_class:
                        continue

                    image_array = np.frombuffer(camera_image.raw_data, dtype=np.uint8)
                    image_array = image_array.reshape((camera_image.height, camera_image.width, 4))[:, :, :3]
                    image_array = image_array[:, :, ::-1].copy()
                    grayscale_image = Image.fromarray(image_array, mode='RGB').convert('L')
                    image_name = 'frame_{:08d}.png'.format(next_image_index)
                    grayscale_image.save(os.path.join(output_directory, image_name))
                    label_writer.writerow((image_name, class_label, scene_number,
                                           weather_name,
                                           spawn_index, camera_image.frame))
                    labels_file.flush()
                    class_counts[class_label] += 1
                    next_image_index += 1
                    if next_image_index % 100 == 0:
                        print('{}: saved {} frames; class counts {}'.format(
                            split_name, next_image_index, class_counts
                        ))
            finally:
                if camera_sensor is not None:
                    camera_sensor.stop()
                    camera_sensor.destroy()
                for background_vehicle in background_vehicles:
                    background_vehicle.set_autopilot(False, arguments.traffic_manager_port)
                    background_vehicle.destroy()
                ego_vehicle.set_autopilot(False, arguments.traffic_manager_port)
                ego_vehicle.destroy()
                try:
                    traffic_manager.set_synchronous_mode(False)
                except (AttributeError, RuntimeError):
                    pass
                carla_world.apply_settings(original_settings)
            episode_index += 1
    print('{} collection complete: {} clear, {} obstacle frames'.format(
        split_name, class_counts[0], class_counts[1]
    ))
    if class_counts[0] < arguments.samples_per_class or class_counts[1] < arguments.samples_per_class:
        raise RuntimeError('Frame limit reached before both classes reached the requested count')


def main():
    if np is None:
        raise ImportError('NumPy is required; install the CARLA perception dependencies')
    arguments = parse_arguments()
    random.seed(arguments.seed)
    client = carla.Client(arguments.host, arguments.port)
    client.set_timeout(arguments.timeout)
    split_names = ('train', 'val', 'test') if arguments.split == 'all' else (arguments.split,)
    for split_index, split_name in enumerate(split_names):
        collect_split(client, split_name, arguments, arguments.seed + split_index * 100003)


if __name__ == '__main__':
    main()
