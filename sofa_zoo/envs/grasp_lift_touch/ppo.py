import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import (
    CollisionEffect, GraspLiftTouchEnv, Phase, RenderMode, ObservationType, ActionType
)
from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS
import wandb
from wandb.integration.sb3 import WandbCallback
from enum import Enum


class SaveCheckpointCallback(BaseCallback):
    """Saves model and VecNormalize stats together at regular intervals."""
    
    def __init__(self, save_path: str, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save model with timestep
            model_path = os.path.join(self.save_path, f"model_{self.num_timesteps}.zip")
            self.model.save(model_path)
            
            # Save VecNormalize - use model.get_env() which has the full wrapper stack
            # The VecNormalize is inside the wrapper stack and will be saved correctly
            vecnorm_path = os.path.join(self.save_path, f"vecnormalize_{self.num_timesteps}.pkl")
            self.model.get_env().save(vecnorm_path)
            
            if self.verbose > 0:
                print(f"✓ Checkpoint saved at step {self.num_timesteps}")
                print(f"  Model: {model_path}")
                print(f"  VecNormalize: {vecnorm_path}")
        return True


if __name__ == "__main__":

    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    parameters = ["STATE", ["GRASP", "DONE"]]
    start_name = parameters[1][0]
    end_name = parameters[1][1]
    start_phase = Phase[start_name]
    end_phase = Phase[end_name]

    observation_type = ObservationType[parameters[0]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

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
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
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
        config={
            "observation_type": observation_type.name,
            "continuous_actions": continuous_actions,
            "start_phase": start_phase.name,
            "end_phase": end_phase.name,
            "total_timesteps": config["total_timesteps"],
            **make_wandb_safe(config),
            **make_wandb_safe(ppo_kwargs),
            **make_wandb_safe(env_kwargs),
        },
        sync_tensorboard=True,
        monitor_gym=True,
        name=f"PPO_{observation_type.name}_cont={continuous_actions}",
    )

    # Create save directory with run ID
    save_path = f"models/{wandb.run.id}"
    os.makedirs(save_path, exist_ok=True)

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

    # Save checkpoint every 80000 steps (model + vecnormalize together)
    checkpoint_callback = SaveCheckpointCallback(
        save_path=save_path,
        save_freq=5000,
        verbose=1,
    )

    combined_callback = CallbackList([callback, checkpoint_callback])

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=combined_callback,
        tb_log_name=f"PPO_{observation_type.name}_{continuous_actions=}_start={start_phase.name}_end={end_phase.name}",
    )

    # Save final checkpoint
    model.save(f"{save_path}/final_model.zip")
    model.get_env().save(f"{save_path}/final_vecnormalize.pkl")
    print(f"✓ Training complete! Final checkpoint saved to {save_path}")
    
    wandb.finish()