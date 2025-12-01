# Code mostly taken from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py
from __future__ import annotations

import datetime
import functools
import logging
import multiprocessing
import os
import time
from copy import deepcopy
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import optuna

# from torch.utils.tensorboard import SummaryWriter
import wandb
from sympol.rllib_port.mlp.state_action_dt import fit_stateActionDT
from sympol.rllib_port.sympol_setup import SympolSetup
from sympol.rllib_port.core.sympol_module import SympolPPOModule

from sympol.utils import (
    ActorTrainState,
    EpisodeStatistics,
    Storage,
    TrainState,
    make_training_env,
)
from sympol.utils.evaluation import evaluate_agent
from sympol.utils.ppo import compute_gae, update_ppo
from sympol.utils.rollout import create_rollout_function, update_buffer_and_rollout_size
from sympol.utils.type_guard import is_stateActionDT

if TYPE_CHECKING:
    from sympol.config_types.args_types import SympolCLIArgs
    from sympol.rllib_port.extended_args import SympolArgumentParser
    from sympol.utils.rollout import RolloutCallableType

logger = logging.getLogger(__name__)

# os.environ['MUJOCO_GL'] = 'egl'


# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"
# os.environ["XLA_FLAGS"] = "--xla_dump_to=~/tmp/foo"

# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
# os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
# os.environ["TF_CUDNN DETERMINISTIC"] = "1"


def train_agent(
    setup: SympolSetup, trial: Optional[optuna.Trial] = None, queue: Optional[multiprocessing.Queue] = None
):
    start_time = time.time()
    args: SympolArgumentParser = deepcopy(setup.args)  # type: ignore
    assert args.seed is not None

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if trial is not None:
        timestamp = "TRIAL_NO" + str(trial.number) + "_" + timestamp

        args.__dict__.update(setup.create_param_space(trial))

        n_steps: int = args.n_steps

    elif not args.use_best_config:
        if False:
            if args.env_id == "Hopper-v4":
                n_steps = 512
                args.n_envs = 2
            elif args.env_id == "CartPole-v1":
                n_steps = 32
                args.n_envs = 8
            elif args.env_id == "Pendulum-v1":
                n_steps = 1024
                args.n_envs = 4
            elif args.env_id == "BipedalWalker-v3":
                n_steps = 512
                args.n_envs = 16
            elif args.env_id == "LunarLander-v2":
                n_steps = 512
                args.n_envs = 8
        else:
            n_steps = 512
            args.n_envs = 8
    else:
        n_steps = args.n_steps

    if args.dynamic_buffer:
        n_steps = max(16, n_steps // 8)

    accumulate_gradients_every = args.accumulate_gradients_every
    accumulate_gradients_every_initial = accumulate_gradients_every

    initial_steps = n_steps
    # these parameters are defined dynamically
    batch_size = int(args.n_envs * n_steps)
    # minibatch_size = int(batch_size // args.n_minibatches)
    minibatch_size: int = args.minibatch_size
    while batch_size // minibatch_size < 2:
        minibatch_size = minibatch_size // 2
    n_iterations = args.total_steps // batch_size
    # eval_freq = max(args.eval_freq // batch_size, 1)

    trial_scores = []
    for random_trial_number in range(1, args.random_trials + 1):
        # region trial setup

        if args.actor == "sympol":
            model_identifier = "-".join([str(args.depth), str(args.n_estimators), str(args.seed)])
        elif args.actor != "mlp":
            model_identifier = "-".join([str(args.depth), str(args.seed)])
        else:
            model_identifier = str(args.seed)

        run_name = "-".join(
            [args.run_name, args.actor, str(np.round(args.learning_rate_actor, 6)), model_identifier, timestamp]
        )

        group_name = run_name
        run_name = run_name + "_" + str(random_trial_number)

        # region build env; calls os.fork
        envs, obs_dim, action_dim, action_indices = make_training_env(args)
        # endregion

        if args.track:
            wandb_run = wandb.init(
                project=f"{args.exp_name}_{args.env_id}",
                group=group_name,
                tags=[args.run_name],
                # sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )

        module = SympolPPOModule(
            observation_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            # FIXME: args should be SympolCLIArgs to assure jax compatible hashing!
            model_config=asdict(args),
        )
        module.setup()

        new_critic = module.vf
        critic = new_critic.model
        critic_state: TrainState = module.states["critic"]
        new_actor = module.pi
        actor = new_actor.model
        actor_state: ActorTrainState = module.states["actor"]

        lr_scheduler = optax.contrib.reduce_on_plateau(patience=3, factor=0.5)
        lr_scheduler_state = lr_scheduler.init(actor_state.params)

        episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
            returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        )

        # endregion
        global_step = 0

        # Seeds
        env_seed = args.seed + (random_trial_number * 100)
        if True:
            seed_training = args.seed + (random_trial_number * 100)
        else:
            seed_training = args.seed
        key = jax.random.PRNGKey(seed_training)

        next_obs, _ = envs.reset(seed=env_seed)
        next_done = np.zeros(args.n_envs).astype(bool)

        # hyperparameters = {key: value for key, value in vars(args).items()}

        # Save hyperparameters to wandb
        # wandb.config.update(hyperparameters)
        avg_score_list = []
        iteration = 1
        last_eval = 0
        n_steps_old = 0

        avg_episodic_return_list = []
        total_time_cleaned = 0

        if args.total_steps == "auto":
            args.total_steps = 1_000_000
        while global_step < args.total_steps:
            # for iteration in range(1, n_iterations + 1):
            wandb_log = {}
            # ALGO Logic: Storage setup
            # increase_index = global_step // (args.total_steps // len(increase_factor_list))
            # region: update buffer and rollout size; create new rollout function
            rollout: RolloutCallableType
            if args.dynamic_buffer or args.dynamic_batch:
                batch_size, accumulate_gradients_every, n_steps = update_buffer_and_rollout_size(
                    total_steps=args.total_steps,
                    dynamic_buffer=args.dynamic_buffer,
                    dynamic_batch=args.dynamic_batch if hasattr(args, "dynamic_batch") else not args.static_batch,
                    n_envs=args.n_envs,
                    initial_steps=initial_steps,
                    global_step=global_step,
                    accumulate_gradients_every_initial=accumulate_gradients_every_initial,
                )

                if n_steps != n_steps_old:
                    # compute_gae  = create_compute_gae(n_steps)
                    # update_ppo = create_update_ppo(batch_size, minibatch_size, accumulate_gradients_every)
                    rollout = create_rollout_function(
                        n_steps, envs, args=args, actor=actor, critic=critic, action_indices=action_indices
                    )
                    n_steps_old = n_steps
            elif global_step == 0:
                # compute_gae  = create_compute_gae(n_steps)
                # update_ppo = create_update_ppo(batch_size, minibatch_size, accumulate_gradients_every)
                rollout = create_rollout_function(
                    n_steps, envs, args=args, actor=actor, critic=critic, action_indices=action_indices
                )
            current_eval = global_step // args.eval_freq
            start_time_cleaned = time.time()

            if TYPE_CHECKING:
                assert envs.single_observation_space.shape is not None
                assert envs.single_action_space.shape is not None
            storage = Storage(
                obs=jnp.zeros((n_steps, args.n_envs, *envs.single_observation_space.shape)),
                actions=jnp.zeros((n_steps, args.n_envs, *envs.single_action_space.shape), dtype=jnp.int32),
                logprobs=jnp.zeros((n_steps, args.n_envs)),
                dones=jnp.zeros((n_steps, args.n_envs)),
                values=jnp.zeros((n_steps, args.n_envs)),
                advantages=jnp.zeros((n_steps, args.n_envs)),
                returns=jnp.zeros((n_steps, args.n_envs)),
                rewards=jnp.zeros((n_steps, args.n_envs)),
            )
            actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step = rollout(  # pyright: ignore[reportPossiblyUnboundVariable]
                actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step
            )
            # print_values("after rollout")
            storage = compute_gae(critic_state, next_obs, next_done, storage, critic=critic, args=args)
            # print_values("with gae")
            actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
                actor_state,
                critic_state,
                storage,
                key,
                accumulate_gradients_every,
                minibatch_size=minibatch_size,
                n_update_epochs=args.n_update_epochs,
                args=args,
                actor=actor,
                critic=critic,
                actor_state_indices=actor_state.indices,
            )

            elapsed_time_cleaned = time.time() - start_time_cleaned
            total_time_cleaned += elapsed_time_cleaned

            avg_episodic_return = np.mean(np.array(episode_stats.returned_episode_returns))
            avg_episodic_return_list.append(avg_episodic_return)

            # writer.add_scalar("charts/avg_train_episodic_return", avg_episodic_return, global_step)
            # region evaluation, test and reporting
            if iteration == 1 or current_eval > last_eval or global_step + batch_size >= args.total_steps:
                last_eval = current_eval
                render_now = (
                    True if args.render_each_eval else True if global_step + batch_size >= args.total_steps else False
                )

                end_time = time.time()
                elapsed_time = end_time - start_time

                if is_stateActionDT(actor, args):
                    decision_tree = fit_stateActionDT(
                        actor_state,
                        args.env_id,
                        n_episodes=25,
                        name_appendix="",
                        seed=args.seed,
                        envs=envs,
                        action_dim=action_dim,
                        n_steps=n_steps,
                        args=args,
                        actor=actor,
                        storage=storage,
                        action_indices=action_indices,
                    )
                else:
                    decision_tree = None

                score, score_interpretable, node_count = evaluate_agent(
                    actor_state,
                    args.env_id,
                    n_episodes=args.n_eval_episodes,
                    name_appendix="",
                    seed=env_seed,
                    decision_tree=decision_tree,
                    args=args,
                    action_dim=action_dim,
                    run_name=run_name,
                    render_now=render_now,
                    action_indices=action_indices,
                    actor=actor,
                    obs_dim=obs_dim,
                )

                avg_score = np.mean(score).item()
                avg_score_interpretable = np.mean(score_interpretable).item()
                std_score = np.std(score).item()
                std_score_interpretable = np.std(score_interpretable).item()
                # use the negative avg score, since reduce on plataeu normally considers non-decreasing losses as a plataeu,
                # but we have a plataeu when the score is not increasing anymore
                # region reduce learning rate
                if args.reduce_lr:
                    _, lr_scheduler_state = lr_scheduler.update(
                        updates=actor_state.params, state=lr_scheduler_state, value=avg_score
                    )
                    # [-1] is the adamw optimizer, while [0] would be the gradient clipping of the tx.chain
                    if TYPE_CHECKING:
                        lr_scheduler_state = cast("optax.contrib.ReduceLROnPlateauState", lr_scheduler_state)
                        actor_state.opt_state = cast("tuple[tuple[Any, ...] | tuple[()], Any]", actor_state.opt_state)
                    if args.actor != "sympol":
                        actor_state.opt_state[1].hyperparams["learning_rate"] = (
                            args.learning_rate_actor * lr_scheduler_state.scale
                        )
                    else:
                        actor_state.opt_state[1][0]["estimator_weights"][0].hyperparams["learning_rate"] = (
                            args.learning_rate_actor_weights * lr_scheduler_state.scale
                        )
                        actor_state.opt_state[1][0]["split_values"][0].hyperparams["learning_rate"] = (
                            args.learning_rate_actor_split_values * lr_scheduler_state.scale
                        )
                        actor_state.opt_state[1][0]["split_idx_array"][0].hyperparams["learning_rate"] = (
                            args.learning_rate_actor_split_idx_array * lr_scheduler_state.scale
                        )
                        actor_state.opt_state[1][0]["leaf_array"][0].hyperparams["learning_rate"] = (
                            args.learning_rate_actor_leaf_array * lr_scheduler_state.scale
                        )
                        actor_state.opt_state[1][0]["log_std"][0].hyperparams["learning_rate"] = (
                            args.learning_rate_actor_log_std * lr_scheduler_state.scale
                        )
                # endregion reduce learning rate

                end_time = time.time()
                elapsed_time = end_time - start_time
                start_time = end_time
                if args.actor in {"stateActionDT", "d-sdt"}:
                    print(
                        f"global_step={global_step}, avg_eval_episodic_return={avg_score}, avg_eval_episodic_return_discrete={avg_score_interpretable} (Elapsed time: {elapsed_time} seconds)"
                    )
                    if args.track:
                        wandb_log["charts/avg_score"] = avg_score_interpretable
                        wandb_log["charts/avg_score_fully_complexity"] = avg_score
                        wandb_log["charts/std_score"] = std_score_interpretable
                        wandb_log["charts/std_score_fully_complexity"] = std_score
                        wandb_log["charts/score_interpretable_list"] = score_interpretable
                        wandb_log["charts/score_list"] = score
                    avg_score_list.append(avg_score_interpretable)
                else:
                    print(
                        f"global_step={global_step}, avg_eval_episodic_return={avg_score} (Elapsed time: {elapsed_time} seconds)"
                    )
                    if args.track:
                        wandb_log["charts/avg_score"] = avg_score
                        wandb_log["charts/std_score"] = std_score
                        wandb_log["charts/score_list"] = score

                    avg_score_list.append(avg_score)
                if args.track:
                    wandb_log["charts/node_count"] = node_count
                    wandb_log["charts/total_time_cleaned"] = total_time_cleaned

                if global_step + batch_size >= args.total_steps:  # TEST EVAL
                    print("-- Running test eval --")
                    test_seed = 123456
                    if is_stateActionDT(actor, args):  # args.actor == "stateActionDT"
                        decision_tree = fit_stateActionDT(
                            actor_state,
                            args.env_id,
                            n_episodes=25,
                            name_appendix="",
                            seed=args.seed,
                            envs=envs,
                            action_dim=action_dim,
                            n_steps=n_steps,
                            args=args,
                            actor=actor,
                            storage=storage,
                            action_indices=action_indices,
                        )
                    else:
                        decision_tree = None

                    score_test, score_interpretable_test, node_count_test = evaluate_agent(
                        actor_state,
                        args.env_id,
                        n_episodes=args.n_eval_episodes,
                        name_appendix="",
                        decision_tree=decision_tree,
                        seed=test_seed,
                        args=args,
                        action_dim=action_dim,
                        run_name=run_name,
                        render_now=render_now,
                        action_indices=action_indices,
                        actor=actor,
                        obs_dim=obs_dim,
                    )

                    avg_score_test = np.mean(score_test).item()
                    avg_score_interpretable_test = np.mean(score_interpretable_test).item()
                    std_score_test = np.std(score_test).item()
                    std_score_interpretable_test = np.std(score_interpretable_test).item()
                    # use the negative avg score, since reduce on plataeu normally considers non-decreasing losses as a plataeu,
                    # but we have a plataeu when the score is not increasing anymore
                    if args.actor in {"stateActionDT", "d-sdt"}:
                        print(
                            f"global_step={global_step}, avg_eval_episodic_return={avg_score_test}, avg_eval_episodic_return_discrete={avg_score_interpretable_test} (Elapsed time: {elapsed_time} seconds)"
                        )
                        if args.track:
                            wandb_log["charts/avg_score_test"] = avg_score_interpretable_test
                            wandb_log["charts/avg_score_fully_complexity_test"] = avg_score_test
                            wandb_log["charts/std_score_test"] = std_score_interpretable_test
                            wandb_log["charts/std_score_fully_complexity_test"] = std_score_test
                            wandb_log["charts/score_list_test"] = score_interpretable_test
                            wandb_log["charts/score_fully_complexity_list_test"] = score_test

                    else:
                        print(
                            f"global_step={global_step}, avg_eval_episodic_return={avg_score_test} (Elapsed time: {elapsed_time} seconds)"
                        )
                        if args.track:
                            wandb_log["charts/avg_score_test"] = avg_score_test
                            wandb_log["charts/std_score_test"] = std_score
                            wandb_log["charts/score_list_test"] = score_test
                    if args.track:
                        wandb_log["charts/node_count_test"] = node_count_test

                try:
                    complexity_add = 1
                    while False:
                        # Evaluate next complexity level
                        string_list = args.env_id.split("-")
                        complexity_level_new = str(int(string_list[-2][-1]) + complexity_add)
                        string_list[-2] = string_list[-2][:-1] + complexity_level_new
                        env_id_new = "-".join(string_list)
                        avg_score = evaluate_agent(
                            actor_state,
                            env_id_new,
                            n_episodes=args.n_eval_episodes,
                            name_appendix="complexity+" + str(complexity_add),
                        )
                        # [-1] is the adamw optimizer, while [0] would be the gradient clipping of the tx.chain

                        print(
                            f"global_step={global_step}, complexity={complexity_level_new} avg_eval_episodic_return={avg_score}"
                        )
                        # writer.add_scalar("charts/avg_score_complexity" + complexity_level_new, avg_score, global_step)
                        if args.track:
                            wandb_log["charts/avg_score_complexity" + complexity_level_new] = avg_score
                        complexity_add += 1
                except Exception:
                    logger.exception("Error evaluating complexity levels")
            # endregion evaluation, test and reporting

            if args.checkpoint:
                import orbax.checkpoint
                from flax.training import orbax_utils

                end_time = time.time()
                elapsed_time = end_time - start_time
                checkpoint_path = os.path.join(args.path, args.run_name)
                os.makedirs(checkpoint_path, exist_ok=True)
                ckpt = {"sympol": actor_state}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(checkpoint_path, ckpt, save_args=save_args)
                end_time = time.time()
                elapsed_time = end_time - start_time
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            # writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            # writer.add_scalar(
            #    "charts/avg_episodic_length", np.mean(np.array(episode_stats.returned_episode_lengths)), global_step
            # )
            # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            # writer.add_scalar("losses/loss", loss.item(), global_step)
            if args.track:
                wandb_log["charts/global_step"] = global_step
                wandb_log["charts/avg_episodic_return"] = avg_episodic_return
                wandb_log["charts/avg_episodic_return_100"] = np.mean(avg_episodic_return_list[-100:])
                wandb_log["charts/avg_episodic_return_10"] = np.mean(avg_episodic_return_list[-10:])
                wandb_log["charts/avg_episodic_length"] = np.mean(np.array(episode_stats.returned_episode_lengths))
                try:
                    wandb_log["losses/value_loss"] = np.mean(v_loss[-1])  # .item()
                    wandb_log["losses/policy_loss"] = np.mean(pg_loss[-1])  # .item()
                    wandb_log["losses/entropy"] = np.mean(entropy_loss[-1])  # .item()
                    wandb_log["losses/approx_kl"] = np.mean(approx_kl[-1])  # .item()
                    wandb_log["losses/loss"] = np.mean(loss[-1])  # .item()
                except:
                    logger.exception("Error logging losses; likely they are items")
                    wandb_log["losses/value_loss"] = v_loss  # .item()
                    wandb_log["losses/policy_loss"] = pg_loss  # .item()
                    wandb_log["losses/entropy"] = entropy_loss  # .item()
                    wandb_log["losses/approx_kl"] = approx_kl  # .item()
                    wandb_log["losses/loss"] = loss  # .item()
                wandb.log(wandb_log)

            iteration = iteration + 1
        if args.track:
            wandb_run.finish()  # pyright: ignore[reportPossiblyUnboundVariable]
        envs.close()

        # trial_scores.append(np.mean(avg_score_list[-5:]))
        trial_scores.append(avg_score_list[-1])

    if queue is None:
        return np.mean(trial_scores)
    queue.put(np.mean(trial_scores))  # Put the result in the queue
    return None


def multiprocessing_objective_fn(args, trial: optuna.Trial):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=train_agent, args=(args, trial, queue), daemon=True)
    p.start()
    p.join()
    result = queue.get()
    return result


if __name__ == "__main__":
    import socket

    from optuna.storages import RDBStorage  # noqa: F401
    from sqlalchemy import create_engine  # noqa: F401

    setup = SympolSetup(init_param_space=False)
    args = setup.args
    print(args)
    if args.optimize_config:
        # used to save information about trials, delete that if you want to start new trials, e.g. after changing the range
        # of hyperparamters or adding/ removing some hyperparameters
        storage = "sqlite:///hpopt_database_" + socket.gethostname() + ".db"

        # Step 2: Create the engine with the specified timeout
        # engine = create_engine("sqlite:///optuna_database.db", connect_args={'timeout': 30})

        # Step 3: Use this engine to create the Optuna storage
        # storage = RDBStorage("sqlite:///optuna_database_rdb.db")

        study = optuna.create_study(
            direction="maximize",
            storage=storage,
            load_if_exists=True,
            study_name=args.exp_name + "__" + args.env_id + "__" + args.run_name + "__" + args.actor,
        )
        objective_fn = functools.partial(multiprocessing_objective_fn, setup)
        # objective_fn = functools.partial(train_agent, args)
        # wandb only works for n_jobs = 1 ! See README.md for more infos about that
        study.optimize(objective_fn, n_trials=args.n_trials, n_jobs=1)
        # study.optimize(objective_fn, n_trials=args.n_trials, n_jobs=1)

    else:
        # FIXME: args should be SympolCLIArgs to assure jax compatible hashing!
        train_agent(setup, trial=None, queue=None)
