from easydict import EasyDict
from lzero.entry import train_muzero

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
# Number of environments to collect data from
collector_env_num = 8
# Number of episodes to run
n_episode = 8
# Number of environments for evaluation
evaluator_env_num = 3
# Number of simulations to perform
num_simulations = 50  # NOTE: sometimes, 25 is too small for complex environments
# Maximum number of steps per episode
max_steps = 5  # NOTE: max_steps should be at least larger than the optimal episode length
# Ratio of replay buffer updates
replay_ratio = 0.25
# Number of updates per collection of data
update_per_collect = int(collector_env_num * max_steps * replay_ratio)
# Size of each batch for training
batch_size = 256  # NOTE: can be larger
# Maximum number of environment steps
max_env_step = int(5e6)
# Ratio for reanalyzing the data
reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

multi_eqn_muzero_config = dict(
    exp_name=f'data_muzero/multieqn/easy/',
    env=dict(
        env_name='multiEqnEasy_env',  # Changed from LunarLander-v2
        max_steps=max_steps,
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=41,  
            action_space_size=21, 
            model_type='mlp',
            hidden_size_list=[1024, 1024, 1024],
            latent_state_dim=512,  # NOTE: Typically, 512 is sufficient for medium complexity environments, but you can test different configurations
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='not_one_hot',
            res_connection_in_dynamics=True,
            norm_type='LN',
        ),
        # root_dirichlet_alpha=0.3, # NOTE: Typically, we use the default value
        # root_exploration_fraction=0.25,
        td_steps=5,
        num_unroll_steps=5,
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        action_type="fixed_action_space",
        game_segment_length=max_steps,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=0.0001,
        ssl_loss_weight=2,
        grad_clip_value=1.0,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(1e3),
        replay_buffer_size=int(1e6),
        manual_temperature_decay=True,  # NOTE: Use manually decayed temperature: 1 -> 0.5 -> 0.25 
        threshold_training_steps_for_final_temperature=int(1e5),  # NOTE: The number of final training iterations to control temperature. Please refer to [here](https://github.com/opendilab/LightZero/blob/main/lzero/policy/scaling_transform.py#L131).
        policy_entropy_weight=0.05,          # NOTE: Entropy-based exploration; typically, 0.1 is too large, but you can test it
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
multi_eqn_muzero_config = EasyDict(multi_eqn_muzero_config)
main_config = multi_eqn_muzero_config

multi_eqn_muzero_create_config = dict(
    env=dict(
        type='multiEqnEasy_env', 
        import_names=['zoo.custom_envs.equation_solver.env_multi_eqn_easy'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
multi_eqn_muzero_create_config = EasyDict(multi_eqn_muzero_create_config)
create_config = multi_eqn_muzero_create_config

if __name__ == "__main__":
    seed = 14850
    train_muzero([main_config, create_config], seed=seed, model_path=main_config.policy.model_path, max_env_step=max_env_step)