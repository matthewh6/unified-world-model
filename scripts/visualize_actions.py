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
import random
from experiments.uwm.train_robomimic import make_robomimic_env
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment

from datasets.utils.loader import make_distributed_data_loader
from experiments.utils import init_wandb, is_main_process

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dtw_distance(seq1, seq2):
    """
    Calculate Dynamic Time Warping distance between two sequences.
    
    Args:
        seq1: First sequence of shape (T, D)
        seq2: Second sequence of shape (T, D)
    
    Returns:
        DTW distance (float)
    """
    n, m = len(seq1), len(seq2)
    
    # Initialize DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill the DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(seq1[i-1], seq2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                        dtw_matrix[i, j-1],    # deletion
                                        dtw_matrix[i-1, j-1])  # match
    
    return dtw_matrix[n, m]


def plot_eef_trajectories_combined(all_gt_eef_positions, all_conditional_eef_positions, all_marginal_eef_positions, save_path=None):
    """
    Create a 3D visualization of all end-effector trajectories from multiple demos on the same plot.
    
    Args:
        all_gt_eef_positions: List of ground truth eef positions for each demo [(T, 3), ...]
        all_conditional_eef_positions: List of conditional eef positions for each demo [(T, 3), ...]
        all_marginal_eef_positions: List of marginal eef positions for each demo [(T, 3), ...]
        save_path: Optional path to save the plot
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each demo's trajectories with different colors and transparency
    num_demos = len(all_gt_eef_positions)

    color_gt_base = plt.get_cmap('tab10')(0)      # blue
    color_cond_base = plt.get_cmap('tab10')(1)    # orange
    color_marg_base = plt.get_cmap('tab10')(3)    # purple

    for i in range(num_demos):
        gt_pos = all_gt_eef_positions[i]
        cond_pos = all_conditional_eef_positions[i]
        marg_pos = all_marginal_eef_positions[i]

        alpha = 0.7  # transparency

        # Slightly vary the color for each demo, but keep base color distinct
        color_gt = color_gt_base
        color_cond = color_cond_base
        color_marg = color_marg_base

        # Ground truth trajectories: solid blue
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
                color=color_gt, alpha=alpha, linewidth=2.2, linestyle='-')

        # Conditional trajectories: dashed orange
        ax.plot(cond_pos[:, 0], cond_pos[:, 1], cond_pos[:, 2],
                color=color_cond, alpha=alpha, linewidth=2.2, linestyle='--', dashes=(6, 3))

        # Marginal trajectories: dash-dot purple
        ax.plot(marg_pos[:, 0], marg_pos[:, 1], marg_pos[:, 2],
                color=color_marg, alpha=alpha, linewidth=2.2, linestyle='-.')

    # Add legend with sample lines using colorblind-friendly colors and distinct styles
    ax.plot([], [], color=color_gt_base, linestyle='-', linewidth=2.5, label='Ground Truth', alpha=0.9)
    ax.plot([], [], color=color_cond_base, linestyle='--', linewidth=2.5, label='Conditional', alpha=0.9, dashes=(6, 3))
    ax.plot([], [], color=color_marg_base, linestyle=':', linewidth=2.5, label='Marginal', alpha=1.0, marker='o', markersize=8, markerfacecolor=color_marg_base, markeredgecolor='k')

    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'End-Effector Trajectories in 3D Space ({num_demos} Demos)')
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust the viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Make the plot more visually appealing
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

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

    # Visualize all 10 LIBERO-10 environments
    libero_10_data_paths = glob.glob(os.path.join(config.libero_10_data_dir, "*.hdf5"))

    for libero_10_data_path in libero_10_data_paths:
        print(f"Processing {libero_10_data_path}")

        # Make env outside the file block
        env = make_robomimic_env(
            dataset_name=config.dataset.name,
            dataset_path=libero_10_data_path,
            shape_meta=config.dataset.shape_meta,
            obs_horizon=model.obs_encoder.num_frames,
            max_episode_length=config.rollout_length,
            record=True,  # Enable video recording for all demos
        )

        # Lists to store DTW distances for averaging
        all_dtw_conditional = []
        all_dtw_marginal = []
        
        # Lists to store all eef positions for combined plotting
        all_gt_eef_positions = []
        all_conditional_eef_positions = []
        all_marginal_eef_positions = []
        
        # Lists to store video frames for visualization
        all_gt_frames = []
        all_conditional_frames = []
        all_marginal_frames = []

        # Read all demo data from the file
        demo_data = []
        with h5py.File(libero_10_data_path, "r") as f:
            num_demos = len(f["data"])
            for i in range(config.num_samples):
                demo_idx = random.randint(0, num_demos - 1)
                initial_state = f["data"][f"demo_{demo_idx}"]["states"][0]
                actions = f["data"][f"demo_{demo_idx}"]["actions"][:(config.model.action_len * config.model_calls)]
                demo_data.append((initial_state, actions))

        # Process all demos outside the file block
        for i, (initial_state, actions) in enumerate(demo_data):
            # Collect eef positions and video frames for each action distribution
            gt_eef_positions = []
            conditional_eef_positions = []
            marginal_eef_positions = []
            
            gt_frames = []
            conditional_frames = []
            marginal_frames = []

            # Ground truth eef tracking and video
            env.reset_to_state(state=initial_state)
            for action in actions:
                env.step([action])
                img = env.render()
                gt_frames.append(img)
                eef_pos = env.env.get_sim_state()[:3]
                gt_eef_positions.append(eef_pos)
            
            # p(a|o) - conditional action sampling
            current_obs = env.reset_to_state(state=initial_state)
            
            for _ in range(config.model_calls):
                obs_tensor = {
                    k: torch.tensor(v, device=DEVICE)[None] for k, v in current_obs.items()
                }
                
                conditional_action_sample = model.sample_conditional_action(obs_dict=obs_tensor).squeeze().cpu().detach().numpy()
                
                for action in conditional_action_sample:
                    env.step([action])
                    img = env.render()
                    conditional_frames.append(img)
                    # Get eef position
                    eef_pos = env.env.get_sim_state()[:3]
                    conditional_eef_positions.append(eef_pos)
                
                current_obs = env._get_obs()
            
            # p(a) - marginal action sampling
            current_obs = env.reset_to_state(state=initial_state)
            
            for _ in range(config.model_calls):
                obs_tensor = {
                    k: torch.tensor(v, device=DEVICE)[None] for k, v in current_obs.items()
                }
                
                marginal_action_sample = model.sample_marginal_action(obs_dict=obs_tensor).cpu().squeeze().detach().numpy()
                
                for action in marginal_action_sample:
                    env.step([action])
                    img = env.render()
                    marginal_frames.append(img)
                    # Get eef position
                    eef_pos = env.env.get_sim_state()[:3]
                    marginal_eef_positions.append(eef_pos)
                
                current_obs = env._get_obs()

            # Convert to numpy arrays
            gt_eef_positions = np.array(gt_eef_positions)
            conditional_eef_positions = np.array(conditional_eef_positions)
            marginal_eef_positions = np.array(marginal_eef_positions)
            
            # Store for combined plotting
            all_gt_eef_positions.append(gt_eef_positions)
            all_conditional_eef_positions.append(conditional_eef_positions)
            all_marginal_eef_positions.append(marginal_eef_positions)
            
            # Store video frames
            all_gt_frames.append(np.array(gt_frames))
            all_conditional_frames.append(np.array(conditional_frames))
            all_marginal_frames.append(np.array(marginal_frames))
            
            # Calculate DTW distances
            dtw_conditional = dtw_distance(gt_eef_positions, conditional_eef_positions)
            dtw_marginal = dtw_distance(gt_eef_positions, marginal_eef_positions)
            
            all_dtw_conditional.append(dtw_conditional)
            all_dtw_marginal.append(dtw_marginal)

        # Close the environment
        env.close()

        # Create combined 3D plot with all trajectories
        eef_fig = plot_eef_trajectories_combined(all_gt_eef_positions, all_conditional_eef_positions, all_marginal_eef_positions)
        
        # Calculate average DTW distances
        avg_dtw_conditional = np.mean(all_dtw_conditional)
        avg_dtw_marginal = np.mean(all_dtw_marginal)
        std_dtw_conditional = np.std(all_dtw_conditional)
        std_dtw_marginal = np.std(all_dtw_marginal)
        
        # Log everything to wandb
        if wandb.run is not None:
            dataset_name = os.path.basename(libero_10_data_path).replace('.hdf5', '')
            
            # Log combined eef trajectory plot
            wandb.log({
                f"{dataset_name}/eef_trajectory_plot": wandb.Image(eef_fig),
            })
            plt.close(eef_fig)
            
            # Log videos for each demo
            for i in range(config.num_samples):
                gt_video = all_gt_frames[i].transpose(0, 3, 1, 2)[None]
                conditional_video = all_conditional_frames[i].transpose(0, 3, 1, 2)[None]
                marginal_video = all_marginal_frames[i].transpose(0, 3, 1, 2)[None]
                
                wandb.log({
                    f"{dataset_name}/ground_truth_video": wandb.Video(gt_video, fps=10, format="gif"),
                    f"{dataset_name}/conditional_video": wandb.Video(conditional_video, fps=10, format="gif"),
                    f"{dataset_name}/marginal_video": wandb.Video(marginal_video, fps=10, format="gif"),
                })
            
            # Log average DTW distances
            wandb.log({
                f"{dataset_name}/avg_dtw_conditional": avg_dtw_conditional,
                f"{dataset_name}/avg_dtw_marginal": avg_dtw_marginal,
            })


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="visualize_actions_uwm_robomimic.yaml",
)
def main(config):
    # Resolve hydra config
    OmegaConf.resolve(config)

    # Set seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    visualize_action(rank=0, world_size=1, config=config)

if __name__ == "__main__":
    main()
