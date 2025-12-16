import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import (
    CollisionEffect, GraspLiftTouchEnv, Phase, RenderMode, ObservationType, ActionType
)
from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS

import wandb
from enum import Enum


class SaveCheckpointCallback(BaseCallback):
    """Saves model and VecNormalize stats together at regular intervals."""
    
    def __init__(self, save_path: str, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"model_{self.num_timesteps}.zip")
            self.model.save(model_path)
            
            vecnorm_path = os.path.join(self.save_path, f"vecnormalize_{self.num_timesteps}.pkl")
            self.model.get_env().save(vecnorm_path)
            
            if self.verbose > 0:
                print(f"✓ Checkpoint saved at step {self.num_timesteps}")
        return True


if __name__ == "__main__":

    #########################################
    # UPDATE THESE PATHS FOR YOUR CHECKPOINT
    #########################################
    checkpoint_dir = "models/03g7fwdk"  # <-- PUT YOUR RUN ID HERE
    checkpoint_step = 480000              # <-- PUT THE STEP NUMBER HERE
    
    model_path = f"{checkpoint_dir}/model_{checkpoint_step}.zip"
    vecnorm_path = f"{checkpoint_dir}/vecnormalize_{checkpoint_step}.pkl"
    
    # Verify files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize not found: {vecnorm_path}")
    
    print(f"Loading checkpoint from step {checkpoint_step}")
    print(f"  Model: {model_path}")
    print(f"  VecNormalize: {vecnorm_path}")
    #########################################

    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf
    start_phase = Phase.GRASP
    end_phase = Phase.DONE
    observation_type = ObservationType.STATE
    image_based = False

    env_kwargs = {
        "image_shape": (64, 64),
        "observation_type": observation_type,
        "time_step": 0.05,
        "frame_skip": 2,
        "settle_steps": 50,
        "render_mode": RenderMode.HEADLESS,
        "start_in_phase": start_phase,
        "end_in_phase": end_phase,
        "tool_collision_distance": 5.0,
        "goal_tolerance": 5.0,
        "individual_agents": False,
        "individual_rewards": True,
        "action_type": ActionType.CONTINUOUS,
        "on_reset_callbacks": None,
        "reward_amount_dict": {
            Phase.ANY: {
                "collision_cauter_gripper": -0.1,
                "collision_cauter_gallbladder": -0.1,
                "collision_cauter_liver": -0.1,
                "collision_gripper_liver": -0.01,
                "distance_cauter_target": -0.5,
                "delta_distance_cauter_target": -1.0,
                "target_visible": 0.0,
                "gallbladder_is_grasped": 20.0,
                "new_grasp_on_gallbladder": 10.0,
                "lost_grasp_on_gallbladder": -10.0,
                "active_grasping_springs": 0.0,
                "delta_active_grasping_springs": 0.0,
                "gripper_pulls_gallbladder_out": 0.005,
                "dynamic_force_on_gallbladder": -0.003,
                "successful_task": 200.0,
                "failed_task": -0.0,
                "cauter_action_violated_state_limits": -0.0,
                "cauter_action_violated_cartesian_workspace": -0.0,
                "gripper_action_violated_state_limits": -0.0,
                "gripper_action_violated_cartesian_workspace": -0.0,
                "phase_change": 10.0,
                "overlap_gallbladder_liver": -0.1,
                "delta_overlap_gallbladder_liver": -0.01,
            },
            Phase.GRASP: {
                "distance_gripper_graspable_region": -0.2,
                "delta_distance_gripper_graspable_region": -10.0,
            },
            Phase.TOUCH: {
                "cauter_activation_in_target": 0.0,
                "cauter_delta_activation_in_target": 1.0,
                "cauter_touches_target": 0.0,
                "delta_distance_cauter_target": -5.0,
            },
        },
        "collision_punish_mode": CollisionEffect.CONSTANT,
        "losing_grasp_ends_episode": False,
    }

    config = {"max_episode_steps": 400 + 100 * (end_phase.value - start_phase.value), **CONFIG}
    ppo_kwargs = PPO_KWARGS["state_based"]

    info_keywords = [
        "ret_col_cau_gri", "ret_col_cau_gal", "ret_col_cau_liv", "ret_col_gri_liv",
        "ret_cau_act_vio_sta_lim", "ret_cau_act_vio_car_wor",
        "ret_gri_act_vio_sta_lim", "ret_gri_act_vio_car_wor",
        "ret_dis_cau_tar", "ret_del_dis_cau_tar", "ret_cau_tou_tar",
        "ret_cau_act_in_tar", "ret_tar_vis", "ret_dis_gri_gra_reg",
        "ret_del_dis_gri_gra_reg", "ret_gal_is_gra", "ret_new_gra_on_gal",
        "ret_los_gra_on_gal", "ret_act_gra_spr", "ret_del_act_gra_spr",
        "ret_gri_dis_to_tro", "ret_gri_pul_gal_out", "ret_ove_gal_liv",
        "ret_del_ove_gal_liv", "ret_dyn_for_on_gal", "ret_suc_tas",
        "ret_fai_tas", "ret_pha_cha", "ret",
        "distance_cauter_target", "cauter_touches_target",
        "cauter_activation_in_target", "target_visible",
        "distance_gripper_graspable_region", "gripper_distance_to_trocar",
        "gripper_pulls_gallbladder_out", "successful_task", "final_phase",
    ]

    # Step 1: Create fresh env using configure_learning_pipeline
    # This creates the EXACT same wrapper stack as training
    model, callback = configure_learning_pipeline(
        env_class=GraspLiftTouchEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        normalize_observations=True,
        algo_class=PPO,
        algo_kwargs=ppo_kwargs,
        render=False,
        normalize_reward=normalize_reward,
        reward_clip=reward_clip,
    )

    # Step 2: Load the saved VecNormalize stats into the existing wrapper
    import pickle
    with open(vecnorm_path, 'rb') as f:
        saved_env = pickle.load(f)
    
    # Get the current env and find the VecNormalize layer
    current_env = model.get_env()
    
    # The env is: VecFrameStack(VecNormalize(SubprocVecEnv(...)))
    # The saved_env is a VecNormalize with obs_rms directly on it
    # We need to copy stats into the VecNormalize layer inside VecFrameStack
    
    if hasattr(current_env, 'venv') and hasattr(current_env.venv, 'obs_rms'):
        # VecFrameStack wraps VecNormalize - copy from saved_env directly
        current_env.venv.obs_rms = saved_env.obs_rms
        current_env.venv.ret_rms = saved_env.ret_rms
        current_env.venv.training = True
        print(f"✓ Loaded VecNormalize stats (obs_rms shape: {current_env.venv.obs_rms.mean.shape})")
    elif hasattr(current_env, 'obs_rms'):
        # Direct VecNormalize
        current_env.obs_rms = saved_env.obs_rms
        current_env.ret_rms = saved_env.ret_rms
        current_env.training = True
        print(f"✓ Loaded VecNormalize stats (obs_rms shape: {current_env.obs_rms.mean.shape})")
    else:
        raise RuntimeError("Could not find VecNormalize layer in environment stack!")

    # Step 3: Load model weights
    model.set_parameters(model_path)
    print(f"✓ Loaded model weights")

    # Step 4: Setup wandb for continued training
    def make_wandb_safe(config_dict):
        safe_dict = {}
        for key, value in config_dict.items():
            safe_key = key.name if isinstance(key, Enum) else key
            if isinstance(value, Enum):
                safe_dict[safe_key] = value.name
            elif isinstance(value, dict):
                safe_dict[safe_key] = make_wandb_safe(value)
            elif isinstance(value, (list, tuple)):
                safe_dict[safe_key] = [
                    v.name if isinstance(v, Enum) else 
                    make_wandb_safe(v) if isinstance(v, dict) else v 
                    for v in value
                ]
            elif isinstance(value, (str, int, float, bool, type(None))):
                safe_dict[safe_key] = value
            else:
                safe_dict[safe_key] = str(value)
        return safe_dict

    wandb.init(
        project="grasp-lift-touch-PPO-expert",
        name=f"PPO_STATE_continued_{checkpoint_step}",
        config={
            "continued_from": checkpoint_dir,
            "continued_from_step": checkpoint_step,
            **make_wandb_safe(config),
        },
        sync_tensorboard=True,
    )

    save_path = f"models/{wandb.run.id}"
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = SaveCheckpointCallback(
        save_path=save_path,
        save_freq=10000,
        verbose=1,
    )

    combined_callback = CallbackList([callback, checkpoint_callback])

    # Step 5: Continue training
    additional_timesteps = 1_000_000
    
    print(f"\n🚀 Continuing training for {additional_timesteps} more steps...")
    print(f"   Starting from step {checkpoint_step}")
    
    model.learn(
        total_timesteps=additional_timesteps,
        callback=combined_callback,
        reset_num_timesteps=False,  # Continue from where we left off
    )

    # Save final checkpoint
    model.save(f"{save_path}/final_model.zip")
    model.get_env().save(f"{save_path}/final_vecnormalize.pkl")
    print(f"✓ Training complete! Final checkpoint saved to {save_path}")
    
    wandb.finish()