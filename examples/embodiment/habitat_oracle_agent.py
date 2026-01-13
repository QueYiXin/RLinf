"""
Habitat Oracle Agent using ShortestPathFollower

This script implements an Oracle Agent that uses Habitat's built-in ShortestPathFollower
to automatically navigate to the goal position in each episode.

Unlike the debug_habitat_env.py which uses rlinf's HabitatEnv wrapper, this script
directly uses Habitat's native environment API, avoiding multiprocessing complications.
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import habitat
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from rlinf.envs.habitat.venv import HabitatRLEnv

from rlinf.envs.habitat.extensions.utils import observations_to_image
from rlinf.envs.utils import save_rollout_video


def create_oracle_agent(
    config_path: str,
    num_episodes: int = 1,
    save_video: bool = True,
    video_output_dir: str = "private/test_videos",
    video_fps: int = 10,
    save_actions: bool = True,
    actions_output_dir: str = "private/actions",
):
    """
    Create and run an Oracle Agent using ShortestPathFollower.

    Args:
        config_path: Path to Habitat config file
        num_episodes: Number of episodes to run
        save_video: Whether to save video of the episodes
        video_output_dir: Directory to save videos
        video_fps: Frames per second for video
        save_actions: Whether to save action sequences
        actions_output_dir: Directory to save action sequences
    """
    # Load Habitat configuration
    config = get_habitat_config(config_path)

    # Create actions output directory if needed
    if save_actions:
        Path(actions_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Actions will be saved to: {actions_output_dir}")

    # Get success distance from config, default to 0.2 if not found
    # goal_radius = getattr(
    #     config.habitat.task.measurements.success,
    #     "success_distance",
    #     0.2
    # )
    goal_radius = 0.2

    print(f"Goal radius (success distance): {goal_radius}")

    # Create Habitat environment directly using native RLEnv
    # This avoids using rlinf's wrapper and multiprocessing complications
    dataset = habitat.datasets.make_dataset(
        config.habitat.dataset.type,
        config=config.habitat.dataset,
    )
    episode_len = len(dataset.episodes)
    num_episodes = episode_len
    env = HabitatRLEnv(config=config, dataset=dataset)
    env.seed(config.habitat.seed)

    # Initialize ShortestPathFollower
    # ShortestPathFollower needs direct access to the simulator
    follower = ShortestPathFollower(
        env.habitat_env.sim, goal_radius, return_one_hot=False
    )

    # Map action indices to action names for logging
    action_names = {0: "move_forward", 1: "turn_left", 2: "turn_right", 3: "stop"}

    episode_stats = []

    for episode_idx in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {episode_idx + 1}/{num_episodes}")
        

        # Reset environment
        observations = env.reset()
        current_episode = env.habitat_env.current_episode
        print(f'Current Episode_id {current_episode}')
        print(f"{'=' * 60}")
        if current_episode is None:
            print("Warning: Could not get current episode, skipping...")
            continue

        # Get goal position
        if not hasattr(current_episode, "goals") or len(current_episode.goals) == 0:
            print("Warning: Episode has no goals, skipping...")
            continue

        goal_pos = current_episode.goals[0].position
        start_pos = env.habitat_env.sim.get_agent_state().position

        print(f"Start position: {start_pos}")
        print(f"Goal position: {goal_pos}")

        # Episode statistics
        step_count = 0
        total_reward = 0.0
        max_steps = config.habitat.environment.max_episode_steps

        # Action sequence for this episode
        action_sequence = []

        # Video frames for this episode
        video_frames = []

        # Save initial frame after reset
        if save_video:
            metrics = env.habitat_env.get_metrics()
            frame_dict = observations_to_image(observations, metrics)
            frame_parts = []
            if "rgb" in frame_dict:
                frame_parts.append(frame_dict["rgb"])
            if "depth" in frame_dict:
                frame_parts.append(frame_dict["depth"])
            if "top_down_map" in frame_dict:
                frame_parts.append(frame_dict["top_down_map"])
            if frame_parts:
                frame_concat = np.concatenate(frame_parts, axis=1)
                video_frames.append(frame_concat)

        # Run episode
        done = False
        while not done and step_count < max_steps:
            # Get next action from ShortestPathFollower
            best_action = follower.get_next_action(goal_pos)

            if best_action is None:
                print(f"\nStep {step_count}: Reached goal or cannot reach goal")
                # Check if we actually reached the goal
                metrics = env.habitat_env.get_metrics()
                if metrics.get("success", False):
                    print("✓ Successfully reached the goal!")
                else:
                    print("✗ Cannot reach goal (pathfinding failed)")
                break

            # Execute action
            action_name = action_names.get(best_action, f"unknown({best_action})")
            observations, reward, done, info = env.step(best_action)

            step_count += 1
            total_reward += reward

            # Record action information
            action_info = {
                "step": step_count,
                "action": best_action,
            }
            action_sequence.append(action_info)

            # Save video frame if enabled
            if save_video:
                # Convert observations to image
                frame_dict = observations_to_image(observations, info)

                # Concatenate images horizontally: rgb, depth, top_down_map
                frame_parts = []
                if "rgb" in frame_dict:
                    frame_parts.append(frame_dict["rgb"])
                if "depth" in frame_dict:
                    frame_parts.append(frame_dict["depth"])
                if "top_down_map" in frame_dict:
                    frame_parts.append(frame_dict["top_down_map"])

                if frame_parts:
                    frame_concat = np.concatenate(frame_parts, axis=1)
                    video_frames.append(frame_concat)

            # Print progress every 10 steps
            if step_count % 10 == 0:
                metrics = env.habitat_env.get_metrics()
                distance_to_goal = metrics.get("distance_to_goal", float("inf"))
                print(
                    f"Step {step_count}: Action={action_name}, "
                    f"Reward={reward:.3f}, Distance to goal={distance_to_goal:.2f}"
                )

            # Check for success
            metrics = env.habitat_env.get_metrics()
            if metrics.get("success", False):
                print(f"\n✓ Step {step_count}: Successfully reached the goal!")
                done = True

        # Episode summary
        final_metrics = env.habitat_env.get_metrics()
        success = final_metrics.get("success", False)
        spl = final_metrics.get("spl", 0.0)
        distance_to_goal = final_metrics.get("distance_to_goal", float("inf"))

        print(f"\nEpisode {episode_idx + 1} Summary:")
        print(f"  Steps: {step_count}/{max_steps}")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Success: {success}")
        print(f"  SPL (Success weighted by Path Length): {spl:.3f}")
        print(f"  Final distance to goal: {distance_to_goal:.2f}")

        # Save video for this episode
        if save_video and len(video_frames) > 0:
            episode_id = (
                current_episode.episode_id
                if hasattr(current_episode, "episode_id")
                else episode_idx
            )
            video_name = f"episode_{episode_id}"
            try:
                save_rollout_video(
                    video_frames,
                    output_dir=video_output_dir,
                    video_name=video_name,
                    fps=video_fps,
                )
                print(f"  Video saved: {video_output_dir}/{video_name}.mp4")
            except Exception as e:
                print(f"  Warning: Failed to save video: {e}")

        # Save action sequence for this episode
        if save_actions and len(action_sequence) > 0:
            episode_id = (
                current_episode.episode_id
                if hasattr(current_episode, "episode_id")
                else episode_idx
            )

            # Create action data structure
            action_data = {
                "episode_id": episode_id,
                "scene_id": current_episode.scene_id if hasattr(current_episode, "scene_id") else "unknown",
                # "start_position": start_pos.tolist() if isinstance(start_pos, np.ndarray) else list(start_pos),
                # "goal_position": goal_pos.tolist() if isinstance(goal_pos, np.ndarray) else list(goal_pos),
                # "num_steps": step_count,
                # "success": success,
                # "spl": spl,
                # "total_reward": total_reward,
                # "final_distance": distance_to_goal,
                "actions": action_sequence,
            }

            # Save to JSON file
            action_file = Path(actions_output_dir) / f"episode_{episode_id}_actions.json"
            try:
                with open(action_file, "w") as f:
                    json.dump(action_data, f, indent=2)
                print(f"  Actions saved: {action_file}")
            except Exception as e:
                print(f"  Warning: Failed to save actions: {e}")

        episode_stats.append(
            {
                "episode": episode_idx + 1,
                "steps": step_count,
                "success": success,
                "spl": spl,
                "total_reward": total_reward,
                "final_distance": distance_to_goal,
            }
        )

    # # Overall statistics
    # print(f"\n{'=' * 60}")
    # print("Overall Statistics:")
    # print(f"{'=' * 60}")
    # if episode_stats:
    #     success_rate = sum(s["success"] for s in episode_stats) / len(episode_stats)
    #     avg_spl = sum(s["spl"] for s in episode_stats) / len(episode_stats)
    #     avg_steps = sum(s["steps"] for s in episode_stats) / len(episode_stats)
    #     avg_reward = sum(s["total_reward"] for s in episode_stats) / len(episode_stats)

    #     print(f"Episodes run: {len(episode_stats)}")
    #     print(f"Success rate: {success_rate:.2%}")
    #     print(f"Average SPL: {avg_spl:.3f}")
    #     print(f"Average steps: {avg_steps:.1f}")
    #     print(f"Average reward: {avg_reward:.3f}")

    # Clean up
    env.close()

    return episode_stats


def main():
    parser = argparse.ArgumentParser(
        description="Habitat Oracle Agent using ShortestPathFollower"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get(
            "HABITAT_CONFIG_PATH", "/data/qianyx/RLinf/examples/results/private/configs/vlnce_r2r.yaml"
        ),
        help="Path to Habitat config file",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=8, help="Number of episodes to run"
    )
    parser.add_argument(
        "--save-video", action="store_true", default=True, help="Save video of episodes"
    )
    parser.add_argument(
        "--no-save-video",
        dest="save_video",
        action="store_false",
        help="Don't save video",
    )
    parser.add_argument(
        "--video-output-dir",
        type=str,
        default="/data/qianyx/RLinf/examples/results/private/origin_test_videos",
        help="Directory to save videos",
    )
    parser.add_argument(
        "--video-fps", type=int, default=3, help="Frames per second for video"
    )
    parser.add_argument(
        "--save-actions",
        action="store_true",
        default=True,
        help="Save action sequences of episodes",
    )
    parser.add_argument(
        "--no-save-actions",
        dest="save_actions",
        action="store_false",
        help="Don't save action sequences",
    )
    parser.add_argument(
        "--actions-output-dir",
        type=str,
        default="/data/qianyx/RLinf/examples/results/private/actions",
        help="Directory to save action sequences",
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    print(f"Using config: {args.config}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Save video: {args.save_video}")
    if args.save_video:
        print(f"Video output directory: {args.video_output_dir}")
        print(f"Video FPS: {args.video_fps}")
    print(f"Save actions: {args.save_actions}")
    if args.save_actions:
        print(f"Actions output directory: {args.actions_output_dir}")

    # Run Oracle Agent
    stats = create_oracle_agent(
        args.config,
        args.num_episodes,
        save_video=args.save_video,
        video_output_dir=args.video_output_dir,
        video_fps=args.video_fps,
        save_actions=args.save_actions,
        actions_output_dir=args.actions_output_dir,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
