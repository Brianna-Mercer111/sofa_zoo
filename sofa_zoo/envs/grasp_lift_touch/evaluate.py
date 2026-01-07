import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
import pickle

from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import (
    GraspLiftTouchEnv, Phase, RenderMode, ObservationType, ActionType, CollisionEffect
)

if __name__ == "__main__":
    
    # Checkpoint to evaluate
    checkpoint_dir = "models/i9i9du6g"
    checkpoint_step = 2880000  # Latest checkpoint
    
    model_path = f"{checkpoint_dir}/model_{checkpoint_step}.zip"
    vecnorm_path = f"{checkpoint_dir}/vecnormalize_{checkpoint_step}.pkl"
    
    # Same env settings as training
    env_kwargs = {
        "image_shape": (64, 64),
        "observation_type": ObservationType.STATE,
        "time_step": 0.05,
        "frame_skip": 2,
        "settle_steps": 50,
        "render_mode": RenderMode.HEADLESS,  # Change to HEADLESS for faster eval
        "start_in_phase": Phase.GRASP,
        "end_in_phase": Phase.DONE,
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

    # Create environment
    env = DummyVecEnv([lambda: GraspLiftTouchEnv(**env_kwargs)])
    
    # Load VecNormalize stats
    with open(vecnorm_path, 'rb') as f:
        saved_vecnorm = pickle.load(f)
    
    # Wrap with VecNormalize
    env = VecNormalize(env, training=False, norm_reward=False)
    env.obs_rms = saved_vecnorm.obs_rms
    env.ret_rms = saved_vecnorm.ret_rms
    
    # Add frame stacking (4 frames like training)
    env = VecFrameStack(env, n_stack=4)
    
    # Load model
    model = PPO.load(model_path)
    
    print(f"Loaded model from {model_path}")
    print(f"Loaded VecNormalize from {vecnorm_path}")
    
    # Evaluation
    n_episodes = 50
    successes = 0
    episode_rewards = []
    episode_lengths = []
    final_phases = []

    obs = env.reset()
    ep_reward = 0
    ep_length = 0
    ep_count = 0
    
    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward[0]
        ep_length += 1
        
        if done[0]:
            final_phase = info[0].get('final_phase', 0)
            success = (final_phase == 3)
            
            if success:
                successes += 1
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            final_phases.append(final_phase)
            
            ep_count += 1
            print(f"Episode {ep_count}: reward={ep_reward:.1f}, length={ep_length}, phase={final_phase}, success={success}")
            
            ep_reward = 0
            ep_length = 0
            obs = env.reset()

    print(f"\n{'='*50}")
    print(f"RESULTS over {n_episodes} episodes")
    print(f"{'='*50}")
    print(f"Success rate: {successes/n_episodes*100:.1f}%")
    print(f"Mean reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Phase distribution: {dict(zip(*np.unique(final_phases, return_counts=True)))}")
    
    env.close()