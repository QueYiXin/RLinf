import os
import re
import json
import hydra
import numpy as np

from rlinf.config import validate_cfg
from rlinf.envs.habitat.habitat_env import HabitatEnv

# 动作映射表
action_map = {0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right"}

def get_gt_episode_ids(gt_dir_path: str):
    episode_ids = []
    if not os.path.exists(gt_dir_path):
        print(f"Error: 路径 {gt_dir_path} 不存在")
        return []
    for filename in os.listdir(gt_dir_path):
        match = re.search(r'episode_(\d+)_actions\.json', filename)  
        if match:
            episode_id = match.group(1)
            episode_ids.append(int(episode_id))
    episode_ids.sort(key=int) 

    return episode_ids


def load_actions_from_file(episode_ids: list, gt_dir_path: str):
    """
    从文件夹加载 JSON 动作序列
    如果存在episode不在里面,则返回空列表
    若均存在, 则返回对应输入episode_ids的action_list
    """
    action_list = []
    gt_episode_list = get_gt_episode_ids(gt_dir_path)

    for episode_id in episode_ids:
        if episode_id not in gt_episode_list:
            return []
        file_name = f"episode_{episode_id}_actions.json"
        file_path = os.path.join(gt_dir_path, file_name)
        with open(file_path, 'r') as f:
            action_data = json.load(f)
        
        # 提取并映射动作
        actions_per_file = [action_map[action_info['action']] for action_info in action_data['actions']]
        action_list.append(actions_per_file)
        
    return action_list 


class gt_sim_model():
    def __init__(self, episode_ids, gt_dir_path):
        self.episode_ids = episode_ids
        self.gt_dir_path = gt_dir_path
        self.action_list = load_actions_from_file(self.episode_ids, self.gt_dir_path)


def test_habitat_env(cfg):
    # 单轮episode，暂时没有auto_reset
    action_dir_path = "examples/results/private/actions"
    num_envs = 3
    seed_offset = 0
    total_num_processes = 1
    
    max_steps = cfg.env.eval.max_episode_steps
    chunk_size = cfg.actor.model.num_action_chunks
    n_loops = max_steps // chunk_size

    print("="*60);print("Initing Env");print("="*60)
    env = HabitatEnv(
        cfg=cfg.env.eval,
        num_envs=num_envs,
        seed_offset=seed_offset,
        total_num_processes=total_num_processes,
    )
    env.reset()
    print("="*60);print("End init Env");print("="*60)
    print("="*60);print("reading files");print("="*60)
    episode_ids = env.env.get_current_episode_ids()
    print(episode_ids)
    actions_lists = load_actions_from_file(episode_ids, action_dir_path)

    assert actions_lists, f"存在env中的episode在gt中没有数据"
    stop_flag = 0
    min_action_seq = min([len(seq) for seq in actions_lists])
    print("="*60);print("End reading files");print("="*60)
    for i in range(n_loops):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        if end_idx >= min_action_seq:
            stop_flag = 1
            end_idx = min_action_seq
        current_chunk_batch = []

        for env_idx in range(num_envs):
            seq = actions_lists[env_idx]
            chunk = seq[start_idx:end_idx]
            # if stop_flag:
            #     chunk[-1] = "stop"
            current_chunk_batch.append(chunk)

        actions_to_step = np.array(current_chunk_batch)
        print("-"*60)
        print(actions_to_step)
        print("-"*60)

        try:
            env.chunk_step(actions_to_step)
            print(f"Step {i}: [{start_idx} to {end_idx}] executed.")
            if stop_flag:
                break
        except AssertionError:
            print(f"Step {i}: Environment terminated early.")
            break

    # for video_name, video_frames in env.render_images.items():
    #     env.flush_video(video_name, video_frames)
    env.flush_video()


@hydra.main(version_base="1.1", config_path="config", config_name="habitat_r2r_grpo_cma")
def main(cfg):
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    
    test_habitat_env(cfg=cfg)

if __name__ == "__main__":
    main()
