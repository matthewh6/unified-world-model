import glob
import os

import h5py
import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
import wandb
import imageio
from experiments.uwm.train_robomimic import make_robomimic_env

from datasets.utils.loader import make_distributed_data_loader
from experiments.utils import init_wandb, is_main_process

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_actions_3d(ground_truth_action, conditional_action_sample, marginal_action_sample, save_path=None):
    """
    Create a 3D visualization of the actions.
    
    Args:
        ground_truth_action: Ground truth actions (T, 3)
        conditional_action_sample: Conditional action samples (T, 3)
        marginal_action_sample: Marginal action samples (T, 3)
        save_path: Optional path to save the plot
    """
    # Convert to numpy if they're tensors
    if hasattr(ground_truth_action, 'cpu'):
        ground_truth_action = ground_truth_action.detach().cpu().numpy()
    if hasattr(conditional_action_sample, 'cpu'):
        conditional_action_sample = conditional_action_sample.detach().cpu().numpy()
    if hasattr(marginal_action_sample, 'cpu'):
        marginal_action_sample = marginal_action_sample.detach().cpu().numpy()

    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each trajectory
    # Ground truth in blue
    ax.plot(ground_truth_action[:, 0], 
            ground_truth_action[:, 1], 
            ground_truth_action[:, 2], 
            'b-', label='Ground Truth', linewidth=2)
    
    # Conditional in green
    ax.plot(conditional_action_sample[:, 0], 
            conditional_action_sample[:, 1], 
            conditional_action_sample[:, 2], 
            'g--', label='Conditional', linewidth=2)
    
    # Marginal in red
    ax.plot(marginal_action_sample[:, 0], 
            marginal_action_sample[:, 1], 
            marginal_action_sample[:, 2], 
            'r:', label='Marginal', linewidth=2)

    # Add markers at the start and end points
    ax.scatter(ground_truth_action[0, 0], 
              ground_truth_action[0, 1], 
              ground_truth_action[0, 2], 
              c='b', marker='o', s=100, label='Start')
    ax.scatter(ground_truth_action[-1, 0], 
              ground_truth_action[-1, 1], 
              ground_truth_action[-1, 2], 
              c='b', marker='*', s=200, label='End')

    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Action Trajectories in 3D Space')
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add grid
    ax.grid(True)
    
    # Adjust the viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Make the plot more visually appealing
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_action(rank, world_size, config):
    # Initialize wandb
    if is_main_process():
        init_wandb(config, job_type="visualization")
    
    # Load model from checkpoint
    model = instantiate(config.model).to(DEVICE)
    ckpt = torch.load(config.model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"Loaded pretraining checkpoint {config.pretrain_checkpoint_path}, step: {ckpt['step']}"
    )

    # Log all 10 LIBERO-10 environments
    libero_10_data_paths = glob.glob(os.path.join(config.libero_10_data_dir, "*.hdf5"))

    for libero_10_data_path in libero_10_data_paths:
        print(f"Processing {libero_10_data_path}")

        with h5py.File(libero_10_data_path, "r") as f:
            initial_state = f["data"]["demo_0"]["states"][0]
            actions = f["data"]["demo_0"]["actions"][:config.model.action_len] * config.model_calls # take how many the model outputs

        env = make_robomimic_env(
            dataset_name=config.dataset.name,
            dataset_path=libero_10_data_path,
            shape_meta=config.dataset.shape_meta,
            obs_horizon=model.obs_encoder.num_frames,
            max_episode_length=config.rollout_length,
            record=True,
        )

        # Collect video frames for each action distribution
        gt_frames = []
        conditional_frames = []
        marginal_frames = []

        # Ground truth video
        env.reset_to_state(state=initial_state)
        for action in actions:
            env.step([action])
            img = env.render()
            gt_frames.append(img)
        
        # Conditional action
        current_obs = env.reset_to_state(state=initial_state)
        
        for i in range(config.model_calls):
            obs_tensor = {
                k: torch.tensor(v, device=DEVICE)[None] for k, v in current_obs.items()
            }
            
            conditional_action_sample = model.sample_conditional_action(obs_dict=obs_tensor).squeeze().cpu().detach().numpy()
            
            # Execute conditional actions and collect frames
            for action in conditional_action_sample:
                env.step([action])
                img = env.render()
                conditional_frames.append(img)
            
            current_obs = env._get_obs()
        
        # Marginal action
        current_obs = env.reset_to_state(state=initial_state)
        
        for i in range(config.model_calls):
            obs_tensor = {
                k: torch.tensor(v, device=DEVICE)[None] for k, v in current_obs.items()
            }
            
            marginal_action_sample = model.sample_marginal_action(obs_dict=obs_tensor).cpu().squeeze().detach().numpy()
            
            # Execute marginal actions and collect frames
            for action in marginal_action_sample:
                env.step([action])
                img = env.render()
                marginal_frames.append(img)
            
            current_obs = env._get_obs()

        # Log to wandb with panel structure
        if is_main_process():
            dataset_name = os.path.basename(libero_10_data_path).replace('.hdf5', '')
            
            # Convert frames to numpy arrays
            gt_frames = np.array(gt_frames)
            conditional_frames = np.array(conditional_frames)
            marginal_frames = np.array(marginal_frames)
            
            # Convert to wandb video format: (T, H, W, C) -> (N, T, C, H, W)
            gt_video = gt_frames.transpose(0, 3, 1, 2)[None]
            conditional_video = conditional_frames.transpose(0, 3, 1, 2)[None]
            marginal_video = marginal_frames.transpose(0, 3, 1, 2)[None]
            
            # Log individual videos for each panel
            wandb.log({
                f"{dataset_name}/ground_truth_video": wandb.Video(gt_video, fps=10, format="mp4"),
                f"{dataset_name}/conditional_video": wandb.Video(conditional_video, fps=10, format="mp4"),
                f"{dataset_name}/marginal_video": wandb.Video(marginal_video, fps=10, format="mp4"),
            })
            
            # Also save individual videos locally for reference
            # imageio.mimsave(f"gt_{dataset_name}.mp4", gt_frames, fps=10)
            # imageio.mimsave(f"conditional_{dataset_name}.mp4", conditional_frames, fps=10)
            # imageio.mimsave(f"marginal_{dataset_name}.mp4", marginal_frames, fps=10)
        

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="visualize_actions_uwm_robomimic.yaml",
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)

    visualize_action(rank=0, world_size=1, config=config)

if __name__ == "__main__":
    main()
