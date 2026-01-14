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


def load_actions_from_file(episode_ids: list, gt_dir_path: str, gt_episode_list):
    """
    从文件夹加载 JSON 动作序列
    如果存在episode不在里面,则返回空列表
    若均存在, 则返回对应输入episode_ids的action_list
    """
    action_list = []

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
    gt_episode_ids = get_gt_episode_ids(action_dir_path)
    actions_lists = load_actions_from_file(episode_ids, action_dir_path, gt_episode_ids)

    assert actions_lists, f"存在env中的episode在gt中没有数据"
    env_idx_offset = [0]*num_envs
    print("="*60);print("End reading files");print("="*60)
    for i in range(n_loops):
        print("#"*60);print(f"below episodes:{episode_ids}, loop: {i}");print("#"*60)
        current_chunk_batch = []
        stop_envs = [False]*num_envs

        for env_idx in range(num_envs):
            seq = actions_lists[env_idx]
            start_idx = (i - env_idx_offset[env_idx]) * chunk_size
            end_idx = (i + 1 - env_idx_offset[env_idx]) * chunk_size
            chunk = seq[start_idx:end_idx]
            if len(chunk) < chunk_size and "stop" in chunk:
                chunk.extend(["no_op"]*(chunk_size-len(chunk)))
                stop_envs[env_idx] = True
            current_chunk_batch.append(chunk)
            print("+"*60);print(f"step:{start_idx}->{end_idx-1}, actions: {chunk}")

        actions_to_step = np.array(current_chunk_batch)

        try:
            env.chunk_step(actions_to_step)
            if True in stop_envs:
                episode_ids = env.env.get_current_episode_ids()
                for idx in range(len(stop_envs)):
                    if stop_envs[idx]:
                        actions_lists[idx] = load_actions_from_file([episode_ids[idx]], action_dir_path, gt_episode_ids)[0]
                        env_idx_offset[idx] = i+1
                        print("&"*60);print(f"env_{idx} action change to {actions_lists[idx][:5]}");print("&"*60)
            
        except AssertionError:
            print(f"Step {i}: Environment terminated early.")
            break

    for video_name, video_frames in env.render_images.items():
        env.flush_video_alive(video_name, video_frames)
    print(f"running episodes: {episode_ids}")


@hydra.main(version_base="1.1", config_path="config", config_name="habitat_r2r_grpo_cma")
def main(cfg):
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    
    test_habitat_env(cfg=cfg)

if __name__ == "__main__":
    main()
