from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import (
    CollisionEffect, GraspLiftTouchEnv, Phase, RenderMode, ObservationType, ActionType
)
from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS

from wandb.integration.sb3 import WandbCallback
import wandb
import numpy as np


if __name__ == "__main__":

    # Path to your WandB checkpoint
    checkpoint_path = "models/c82oaify/model.zip"
    
    # Same settings as original
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf
    start_phase = Phase.GRASP
    end_phase = Phase.DONE
    observation_type = ObservationType.STATE

    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
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

    # Use the same pipeline to create env with proper wrappers
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

    # Load the checkpoint weights into the model
    model.set_parameters(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Initialize wandb
    wandb.init(
        project="grasp-lift-touch",
        name="PPO_STATE_continued_c82oaify",
        sync_tensorboard=True,
    )

    wandb_callback = WandbCallback(
        model_save_path=f"models/{wandb.run.id}",
        model_save_freq=10000,
        verbose=2,
    )

    combined_callback = CallbackList([callback, wandb_callback])

    # Continue training
    additional_timesteps = 1_000_000

    model.learn(
        total_timesteps=additional_timesteps,
        callback=combined_callback,
        reset_num_timesteps=False,
        tb_log_name=f"PPO_STATE_continued",
    )

    # Save final model
    model.save(f"models/{wandb.run.id}/final_model.zip")
    wandb.finish()

    print("Training complete!")