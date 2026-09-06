"""Evaluate served policies on LIBERO tasks and save rollout results."""

import collections
import dataclasses
import datetime
import enum
import json
import logging
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.spatial.transform import Rotation as R
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class PolicyType(str, enum.Enum):
    """Supported policy serving modes for LIBERO eval."""

    LAP = "LAP"
    LAP_AR = "LAP_AR"
    RESIDUAL = "RESIDUAL"


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    policy_type: PolicyType = PolicyType.LAP
    # Frame description forwarded to the server and rendered into the prompt.
    # Server default is "robot base frame" if omitted.
    frame_description: str = "end-effector frame"

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    control_mode: str = "OSC_POSE"  # Controller type. Options: OSC_POSE, IK_POSE, OSC_POSITION, JOINT_POSITION, etc.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_out_path: str = "data/libero/results"  # Path to save evaluation results

    seed: int = 7  # Random Seed (for reproducibility)


class LiberoEvaluator:
    """Own a benchmark run, its policy client, and accumulated results."""

    def __init__(self, args: Args) -> None:
        self.args = args

    def _setup(self) -> None:
        args = self.args
        # Set random seed
        np.random.seed(args.seed)

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[args.task_suite_name]()
        logging.info(f"Task suite: {args.task_suite_name}")

        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.results_out_path).mkdir(parents=True, exist_ok=True)

        # Initialize results tracking
        self.results = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "task_suite": args.task_suite_name,
                "policy_type": args.policy_type.value,
                "control_mode": args.control_mode,
                "seed": args.seed,
                "num_trials_per_task": args.num_trials_per_task,
                "replan_steps": args.replan_steps,
            },
            "episodes": [],
            "per_task_results": [],
            "summary": {},
        }

        if args.task_suite_name == "libero_spatial":
            self.max_steps = 220  # longest training demo has 193 steps
        elif args.task_suite_name == "libero_object":
            self.max_steps = 280  # longest training demo has 254 steps
        elif args.task_suite_name == "libero_goal":
            self.max_steps = 300  # longest training demo has 270 steps
        elif args.task_suite_name == "libero_10":
            self.max_steps = 520  # longest training demo has 505 steps
        elif args.task_suite_name == "libero_90":
            self.max_steps = 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task suite: {args.task_suite_name}")

        self.client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

        self.total_episodes = 0
        self.total_successes = 0

    def run(self) -> dict:
        """Evaluate the configured suite and write videos and JSON results."""
        self._setup()
        for task_id in tqdm.tqdm(range(self.task_suite.n_tasks)):
            self._run_task(task_id)
        self._save_results()
        return self.results

    def _run_task(self, task_id: int) -> None:
        args = self.args
        task = self.task_suite.get_task(task_id)
        initial_states = self.task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.control_mode)
        task_episodes = task_successes = 0
        try:
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                result = self._run_episode(env, initial_states[episode_idx], task_id, task_description, episode_idx)
                self.results["episodes"].append(result)
                task_episodes += 1
                task_successes += int(result["success"])
                self.total_episodes += 1
                self.total_successes += int(result["success"])
                logging.info("Success: %s", result["success"])
                logging.info("# episodes completed so far: %d", self.total_episodes)
                logging.info(
                    "# successes: %d (%.1f%%)",
                    self.total_successes,
                    self.total_successes / self.total_episodes * 100,
                )
        finally:
            env.close()

        task_success_rate = task_successes / task_episodes if task_episodes else 0.0
        self.results["per_task_results"].append({
            "task_id": task_id,
            "task_description": task_description,
            "num_episodes": task_episodes,
            "num_successes": task_successes,
            "success_rate": task_success_rate,
        })
        logging.info("Current task success rate: %s", task_success_rate)
        if self.total_episodes:
            logging.info("Current total success rate: %s", self.total_successes / self.total_episodes)

    def _run_episode(self, env, initial_state, task_id: int, task_description: str, episode_idx: int) -> dict:
        args = self.args
        logging.info(f"\nTask: {task_description}")

        # Reset environment
        env.reset()
        action_plan = collections.deque()
        reset_residual_plan = True

        # Set initial states
        obs = env.set_init_state(initial_state)

        # Setup
        t = 0
        done = False
        replay_images = []
        wrist_replay_images = []
        episode_start_time = datetime.datetime.now()

        logging.info(f"Starting episode {episode_idx + 1}...")
        while t < self.max_steps + args.num_steps_wait:
            try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img, wrist_img = get_images_from_obs(obs, args.resize_size)

                if not action_plan:
                    # Query model to get action
                    request = obs_to_request(
                        obs,
                        args.policy_type,
                        img,
                        wrist_img,
                        task_description,
                        args.frame_description,
                        reset_residual_plan=reset_residual_plan,
                    )
                    response = self.client.infer(request)
                    reset_residual_plan = False
                    action_plan.extend(self._action_chunk(response, request))

                # Save preprocessed image for replay video
                replay_images.append(img)
                wrist_replay_images.append(wrist_img)

                action = action_plan.popleft()

                # Execute action in environment
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break
                t += 1

            except Exception as e:
                logging.error(f"Caught exception: {e}")
                break

        # Calculate episode duration
        episode_duration = (datetime.datetime.now() - episode_start_time).total_seconds()

        # Record episode results
        episode_result = {
            "task_id": task_id,
            "task_description": task_description,
            "episode_id": episode_idx,
            "global_episode_id": self.total_episodes,
            "success": bool(done),
            "num_steps": t - args.num_steps_wait,
            "total_steps_with_wait": t,
            "duration_seconds": episode_duration,
        }

        # Save a replay video of the episode
        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_")
        if replay_images:
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_wrist_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in wrist_replay_images],
                fps=10,
            )

        return episode_result

    def _action_chunk(self, response: dict, request: dict) -> np.ndarray:
        """Convert a server response into actions for this execution phase."""
        args = self.args
        single_action_or_chunk = np.asarray(response["actions"], dtype=np.float32)
        if args.policy_type == PolicyType.RESIDUAL:
            if single_action_or_chunk.ndim != 2:
                raise ValueError("residual server must return an action chunk")
            server_replan_steps = int(response.get("replan_steps", args.replan_steps))
            if server_replan_steps != args.replan_steps:
                raise ValueError(
                    f"server uses replan_steps={server_replan_steps}; evaluator was configured for {args.replan_steps}"
                )
            action_chunk = single_action_or_chunk
        elif single_action_or_chunk.ndim == 1:
            assert args.policy_type == PolicyType.LAP_AR
            action_chunk = get_action_from_response(
                args.replan_steps, response, request["observation"]["state"]
            )
        else:
            action_chunk = single_action_or_chunk
        action_chunk = invert_and_scale_gripper(action_chunk)
        if args.policy_type == PolicyType.RESIDUAL:
            # The residual server alternates a base prefix with
            # its corrected tail; execute each returned phase.
            return action_chunk
        else:
            assert len(action_chunk) >= args.replan_steps, (
                f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
            )
            return action_chunk[: args.replan_steps]

    def _save_results(self) -> None:
        args = self.args
        # Calculate and save final summary
        overall_success_rate = float(self.total_successes) / float(self.total_episodes) if self.total_episodes > 0 else 0.0
        self.results["summary"] = {
            "total_episodes": self.total_episodes,
            "total_successes": self.total_successes,
            "overall_success_rate": overall_success_rate,
            "num_tasks": self.task_suite.n_tasks,
        }

        # Save results to JSON file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"results_{args.task_suite_name}_{args.policy_type.value}_{timestamp}.json"
        results_path = pathlib.Path(args.results_out_path) / results_filename
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logging.info(f"Total success rate: {overall_success_rate}")
        logging.info(f"Total episodes: {self.total_episodes}")
        logging.info(f"Results saved to: {results_path}")


def eval_libero(args: Args) -> None:
    """CLI-compatible entry point for benchmark evaluation."""
    LiberoEvaluator(args).run()


def _get_libero_env(task, resolution, seed, controller="OSC_POSE"):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "controller": controller,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_images_from_obs(obs, resize_size):
    # NOTE: Image orientation may vary across MuJoCo / LIBERO versions and camera setups.
    # We apply a horizontal flip by default to match our expected convention.
    # If your images look incorrect (e.g., mirrored or upside down), please adjust
    # the flipping here (i.e., obs["agentview_image"][::-1, ::-1] and obs["robot0_eye_in_hand_image"][::-1, ::-1]).
    # img = np.ascontiguousarray(obs["agentview_image"][:, ::-1])
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


    # Same note applies to the wrist (eye-in-hand) camera.
    # wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][:, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

    return img, wrist_img

def obs_to_request(
    obs,
    policy_type: PolicyType,
    img,
    wrist_img,
    task_description: str,
    frame_description: str = "robot base frame",
    reset_residual_plan: bool = False,
):
    # Prepare observations dict
    assert policy_type in (PolicyType.LAP, PolicyType.LAP_AR, PolicyType.RESIDUAL), f"Unsupported policy type: {policy_type}"
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_rot6d = _quat2rot6d(obs["robot0_eef_quat"]).astype(np.float32, copy=False)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    gripper_state = np.clip(gripper_qpos[-2:-1] / 0.04, 0, 1)  # normalize to [0, 1]
    state = np.concatenate((eef_pos, eef_rot6d, gripper_state)).astype(np.float32, copy=False)
    return {
        "observation": {
            "base_0_rgb": img,
            "left_wrist_0_rgb": wrist_img,
            "state": state,
        },
        "prompt": str(task_description),
        "frame_description": frame_description,
        "reset_residual_plan": reset_residual_plan,
    }


def invert_and_scale_gripper(action_chunk):
    action_chunk[:, -1:] = 1 - 2 * action_chunk[:, -1:]
    action_chunk[:, -1:] = np.sign(action_chunk[:, -1:])
    return action_chunk


def _quat2rot6d(quat):
    "Convert quaternion to 6D rotation representation."
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quat must be shape (4,), ordered as [x, y, z, w]")
    rot_matrix = R.from_quat(q).as_matrix()
    # rot6d = rot_matrix[:, :2].flatten()
    rot6d = np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]], axis=0)
    return rot6d


_OSC_POS_OUTPUT_MAX = 0.05  # meters: OSC_POSE scales [-1, 1] input → [-0.05, 0.05] m
_OSC_ROT_OUTPUT_MAX = 0.5  # radians: OSC_POSE scales [-1, 1] input → [-0.5, 0.5] rad


def get_action_from_response(replan_steps, response, state):
    action = np.asarray(response["actions"])
    grip_action = action[-1]

    # Policy outputs real-world deltas (meters, radians). LIBERO controller expects normalized
    # [-1, 1] inputs which it scales by output_max. Divide total delta evenly across
    # replan_steps (equivalent to uniform SLERP for rotation).

    # Position: normalize by OSC output_max (0.05 m), split across steps
    pos_per_step = (action[:3] / _OSC_POS_OUTPUT_MAX) / replan_steps
    pos_actions = np.tile(pos_per_step, (replan_steps, 1))

    # Rotation: convert delta extrinsic Euler XYZ → axis-angle, normalize by OSC
    # output_max (0.5 rad), split across steps
    delta_rotvec = R.from_euler("xyz", action[3:6]).as_rotvec()
    rot_per_step = (delta_rotvec / _OSC_ROT_OUTPUT_MAX) / replan_steps
    rot_actions = np.tile(rot_per_step, (replan_steps, 1))

    grip_vals = np.full((replan_steps, 1), grip_action)
    return np.concatenate([pos_actions, rot_actions, grip_vals], axis=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
