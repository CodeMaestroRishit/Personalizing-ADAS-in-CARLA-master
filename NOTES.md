# Step 0: repository and integration notes

## Current structure

- `agents/navigation/` contains the roaming, basic, and learning driving agents and their
  local planning/control code.
- `agents/learning/` contains the existing GMM personalization model. This work should
  leave that behavior and the contents of `agents/` intact.
- `environment/ego_vehicle.py` creates the CARLA client, world wrapper, ego vehicle,
  sensors, scene actors, and the pygame control loop.

## Ego vehicle and sensor attachment

`game_loop()` creates a `World` using the parsed spawn location, vehicle blueprint filter,
and scene number. `World.restart()` picks a matching vehicle blueprint, sets its role to
`ego`, and calls `world.try_spawn_actor()` at the configured location (default
`[50, 7.5, 0.5]`). If occupied, spawning fails and the loop returns.

Immediately after a successful ego spawn, `World.restart()` constructs `CameraSet(self.player,
self.hud)`, then the collision and GNSS sensors and radar sensors. This is the existing
sensor attachment point. The camera implementation is in `environment/sensors.py`; a
perception camera can be added there or alongside the existing camera and registered in
`World._ego_list` so `World.destroy()` cleans it up. The sensor objects are attached to the
ego actor, while their relative mount transforms belong in their sensor constructors.

## Control loop and braking hook

In `game_loop()`, the pygame loop waits for a CARLA tick, updates/renders the world, and then
calls `world.agent.run_step(debug=True)`. That returns the proposed `carla.VehicleControl`.
When `world.autopilot_mode` is enabled (the default), the loop applies that control with
`world.player.apply_control(control)`. Here “autopilot” means this repository's selected
navigation agent; it does not call CARLA's built-in `vehicle.set_autopilot()` for the ego.

The minimal perception-driven braking hook is between `run_step()` and `apply_control()`.
If a model prediction crosses the configured threshold, modify that returned control there
(for example, set throttle to zero and raise brake) so braking follows the existing control
path and does not replace the navigation or GMM personalization agents. When manual mode is
active, that agent control is not applied, so the integration should retain that behavior.

## CARLA Python ABI and model process

The repository searches for a CARLA egg whose filename matches the running interpreter's
major/minor Python version. CARLA's 0.9.8 FAQ explains that the egg filename encodes the
Python version and that the client must use the matching interpreter. The CARLA 0.9.8 Linux
release is commonly distributed with `carla-0.9.8-py3.5-linux-x86_64.egg`; verify the actual
egg in the user's downloaded release before choosing the runtime, since that archive is not
present in this checkout. Ubuntu 18.04's usual system Python is 3.6, so an egg limited to
3.5 requires a Python 3.5 client environment.

The planned PyTorch/snnTorch stack is a poor fit for that legacy CARLA Python ABI. To avoid
pinning the perception implementation to a Python/PyTorch version that may no longer support
Python 3.5, the recommended live integration is a separate model process under a supported
modern Python. The CARLA client can send each grayscale frame to a local inference process
over a request/response socket and receive the obstacle probability (and inference timing).
This keeps the CARLA client and its egg in the compatible legacy interpreter while the
model/checkpoint dependencies live in the modern environment. The offline training and
evaluation scripts can use that modern environment directly.

References checked: [CARLA 0.9.8 FAQ](https://carla.readthedocs.io/en/0.9.8/build_faq/)
and [CARLA 0.9.8 release notes](https://carla.org/2020/03/09/release-0.9.8/).
