import datetime
from typing import Dict, NamedTuple, List, Any, Optional, Callable, Set
import cloudpickle
import enum
import time
import pprint

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import (
    UnityCommunicationException,
    UnityTimeOutException,
    UnityEnvironmentException,
    UnityCommunicatorStoppedException,
)
from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection
from queue import Empty as EmptyQueueException
from mlagents_envs.base_env import BaseEnv, BehaviorName, BehaviorSpec
from mlagents_envs import logging_util
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep, AllStepResult
from mlagents.trainers.settings import TrainerSettings
from mlagents_envs.timers import (
    TimerNode,
    timed,
    hierarchical_timer,
    reset_timers,
    get_timer_root,
)
from mlagents.trainers.settings import  RunOptions
from mlagents.trainers.action_info import ActionInfo
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
    EngineConfig,
)
from mlagents_envs.side_channel.stats_side_channel import (
    EnvironmentStats,
    StatsSideChannel,
)

from mlagents_envs.side_channel.side_channel import SideChannel


logger = logging_util.get_logger(__name__)
WORKER_SHUTDOWN_TIMEOUT_S = 10
pp = pprint.PrettyPrinter(indent=4)

class EnvironmentCommand(enum.Enum):
    STEP = 1
    BEHAVIOR_SPECS = 2
    ENVIRONMENT_PARAMETERS = 3
    RESET = 4
    CLOSE = 5
    ENV_EXITED = 6
    CLOSED = 7
    TRAINING_STARTED = 8


class EnvironmentRequest(NamedTuple):
    cmd: EnvironmentCommand
    payload: Any = None


class EnvironmentResponse(NamedTuple):
    cmd: EnvironmentCommand
    worker_id: int
    payload: Any


class StepResponse(NamedTuple):
    all_step_result: AllStepResult
    timer_root: Optional[TimerNode]
    environment_stats: EnvironmentStats


class UnityEnvWorker:
    def __init__(self, worker_id: int, conn: Connection, process : Process):
        self.process = process
        self.worker_id = worker_id
        self.conn = conn
        self.previous_step: EnvironmentStep = EnvironmentStep.empty(worker_id)
        self.previous_all_action_info: ActionInfo = None
        self.waiting = False
        self.closed = False

    def send(self, cmd: EnvironmentCommand, payload: Any = None) -> None:
        try:
            req = EnvironmentRequest(cmd, payload)
            self.conn.send(req)
        except (BrokenPipeError, EOFError):
            raise UnityCommunicationException("UnityEnvironment worker: send failed.")

    def recv(self) -> EnvironmentResponse:
        try:
            response: EnvironmentResponse = self.conn.recv()
            if response.cmd == EnvironmentCommand.ENV_EXITED:
                env_exception: Exception = response.payload
                raise env_exception
            return response
        except (BrokenPipeError, EOFError):
            raise UnityCommunicationException("UnityEnvironment worker: recv failed.")

    def request_close(self):
        try:
            self.conn.send(EnvironmentRequest(EnvironmentCommand.CLOSE))
        except (BrokenPipeError, EOFError):
            logger.debug(
                f"UnityEnvWorker {self.worker_id} got exception trying to close."
            )
            pass


def worker(
    parent_conn: Connection,
    step_queue: Queue,
    pickled_env_factory: str,
    worker_id: int,
    run_options: RunOptions,
    log_level: int = logging_util.INFO,
) -> None:

    env_factory: Callable[
        [int, List[SideChannel]], UnityEnvironment
    ] = cloudpickle.loads(pickled_env_factory)
    env_parameters = EnvironmentParametersChannel()

    engine_config = EngineConfig(
        width=run_options.engine_settings.width,
        height=run_options.engine_settings.height,
        quality_level=run_options.engine_settings.quality_level,
        time_scale=run_options.engine_settings.time_scale,
        target_frame_rate=run_options.engine_settings.target_frame_rate,
        capture_frame_rate=run_options.engine_settings.capture_frame_rate,
    )
    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration(engine_config)

    stats_channel = StatsSideChannel()
    env: UnityEnvironment = None
    # Set log level. On some platforms, the logger isn't common with the
    # main process, so we need to set it again.
    logging_util.set_log_level(log_level)

    def _send_response(cmd_name: EnvironmentCommand, payload: Any) -> None:
        parent_conn.send(EnvironmentResponse(cmd_name, worker_id, payload))

    def _generate_all_results() -> AllStepResult:
        return env.get_steps()

    try:
        side_channels = [env_parameters, engine_configuration_channel, stats_channel]
        env = env_factory(worker_id, side_channels) # Crée un UnityEnv à partir de la env_factory

        while True:
            req: EnvironmentRequest = parent_conn.recv()
            if req.cmd == EnvironmentCommand.STEP:
                action_info = req.payload  # Payload contient l'action à appliquer à l'env
                env.set_actions(action_info.env_action)  # retirer le "behavior"

                env.step()  # Step the environment according to actions set in "env.set_actions()"
                all_step_result = _generate_all_results()
                # The timers in this process are independent from all the processes and the "main" process
                # So after we send back the root timer, we can safely clear them.
                # Note that we could randomly return timers a fraction of the time if we wanted to reduce
                # the data transferred.
                # TODO get gauges from the workers and merge them in the main process too.
                env_stats = stats_channel.get_and_reset_stats()
                step_response = StepResponse(
                    all_step_result, get_timer_root(), env_stats
                )
                step_queue.put(
                    EnvironmentResponse(
                        EnvironmentCommand.STEP, worker_id, step_response
                    )
                )
                reset_timers()
            elif req.cmd == EnvironmentCommand.BEHAVIOR_SPECS:
                _send_response(EnvironmentCommand.BEHAVIOR_SPECS, env.behavior_specs)
            elif req.cmd == EnvironmentCommand.RESET:
                # c'est le env.reset qui génère le behavior
                env.reset()
                all_step_result = _generate_all_results()
                print(all_step_result)
                _send_response(EnvironmentCommand.RESET, all_step_result)
            elif req.cmd == EnvironmentCommand.CLOSE:
                break
    except (
        KeyboardInterrupt,
        UnityCommunicationException,
        UnityTimeOutException,
        UnityEnvironmentException,
        UnityCommunicatorStoppedException,
    ) as ex:
        logger.debug(f"UnityEnvironment worker {worker_id}: environment stopping.")
        step_queue.put(
            EnvironmentResponse(EnvironmentCommand.ENV_EXITED, worker_id, ex)
        )
        _send_response(EnvironmentCommand.ENV_EXITED, ex)
    except Exception as ex:
        logger.exception(
            f"UnityEnvironment worker {worker_id}: environment raised an unexpected exception."
        )
        step_queue.put(
            EnvironmentResponse(EnvironmentCommand.ENV_EXITED, worker_id, ex)
        )
        _send_response(EnvironmentCommand.ENV_EXITED, ex)
    finally:
        logger.debug(f"UnityEnvironment worker {worker_id} closing.")
        if env is not None:
            env.close()
        logger.debug(f"UnityEnvironment worker {worker_id} done.")
        parent_conn.close()
        step_queue.put(EnvironmentResponse(EnvironmentCommand.CLOSED, worker_id, None))
        step_queue.close()


class SubprocessEnvManager(EnvManager):
    """ Démarrre un nombre de process égale au nombre n_env. Ces process utilisent Pipe() qui permet
        la communication duplex enter parent et child process. Les process ont pour target la fonction
        worker(). La fonction qui fait ça : create_worker() renvoie un UnityEnvWorker() qui peut rcv 
        et send des EnvRequest.

        Cette classe sert avant tout à gérer les différents process pour step, reset, exit etc...
    """
    def __init__(
        self,
        env_factory: Callable[[int, List[SideChannel]], BaseEnv],
        run_options: RunOptions,
        n_env: int = 1,
    ):
        super().__init__()
        self.env_workers: List[UnityEnvWorker] = []
        self.step_queue: Queue = Queue()
        self.workers_alive = 0
        self.env_factory = env_factory
        self.run_options = run_options
        self.env_parameters: Optional[Dict] = None

        # Each worker is correlated with a list of times they restarted within the last time period.
        self.recent_restart_timestamps: List[List[datetime.datetime]] = [
            [] for _ in range(n_env)
        ]
        self.restart_counts: List[int] = [0] * n_env

        for worker_idx in range(n_env):
            self.env_workers.append(self.create_worker(worker_idx, self.step_queue, env_factory, run_options))
            self.workers_alive += 1

    @staticmethod
    def create_worker(
        worker_id: int,
        step_queue: Queue,
        env_factory: Callable[[int, List[SideChannel]], BaseEnv],
        run_options: RunOptions,
    ) -> UnityEnvWorker:

        parent_conn, child_conn = Pipe()

        # Need to use cloudpickle for the env factory function since function objects aren't picklable
        # on Windows as of Python 3.6.
        pickled_env_factory = cloudpickle.dumps(env_factory)
        child_process = Process(
            target=worker,
            args=(
                child_conn,
                step_queue,
                pickled_env_factory,
                worker_id,
                run_options,
                logger.level,
            ),
        )
        child_process.start()
        return UnityEnvWorker(worker_id, parent_conn,child_process)

    def _queue_steps(self) -> None:
        for env_worker in self.env_workers:
            if not env_worker.waiting:
                # get_action based on DecisionStep from env_workers
                env_action_info = self._take_step(env_worker.previous_step) 
                # sauvegarde les actions de la step precédante ?
                env_worker.previous_all_action_info = env_action_info
                # Envoie l'info aux workers de step le UnityEnv
                env_worker.send(EnvironmentCommand.STEP, env_action_info)
                env_worker.waiting = True

    def _restart_failed_workers(self, first_failure: EnvironmentResponse) -> None:
        if first_failure.cmd != EnvironmentCommand.ENV_EXITED:
            return
        # Drain the step queue to make sure all workers are paused and we have found all concurrent errors.
        # Pausing all training is needed since we need to reset all pending training steps as they could be corrupted.
        other_failures: Dict[int, Exception] = self._drain_step_queue()
        # TODO: Once we use python 3.9 switch to using the | operator to combine dicts.
        failures: Dict[int, Exception] = {
            **{first_failure.worker_id: first_failure.payload},
            **other_failures,
        }
        for worker_id, ex in failures.items():
            self._assert_worker_can_restart(worker_id, ex)
            logger.warning(f"Restarting worker[{worker_id}] after '{ex}'")
            self.recent_restart_timestamps[worker_id].append(datetime.datetime.now())
            self.restart_counts[worker_id] += 1
            self.env_workers[worker_id] = self.create_worker(
                worker_id, self.step_queue, self.env_factory, self.run_options
            )
        # The restarts were successful, clear all the existing training trajectories so we don't use corrupted or
        # outdated data.
        self.reset(self.env_parameters)

    def _drain_step_queue(self) -> Dict[int, Exception]:
        """
        Drains all steps out of the step queue and returns all exceptions from crashed workers.
        This will effectively pause all workers so that they won't do anything until _queue_steps is called.
        """
        all_failures = {}
        workers_still_pending = {w.worker_id for w in self.env_workers if w.waiting}
        deadline = datetime.datetime.now() + datetime.timedelta(minutes=1)
        while workers_still_pending and deadline > datetime.datetime.now():
            try:
                while True:
                    step: EnvironmentResponse = self.step_queue.get_nowait()
                    if step.cmd == EnvironmentCommand.ENV_EXITED:
                        workers_still_pending.add(step.worker_id)
                        all_failures[step.worker_id] = step.payload
                    else:
                        workers_still_pending.remove(step.worker_id)
                        self.env_workers[step.worker_id].waiting = False
            except EmptyQueueException:
                pass
        if deadline < datetime.datetime.now():
            still_waiting = {w.worker_id for w in self.env_workers if w.waiting}
            raise TimeoutError(f"Workers {still_waiting} stuck in waiting state")
        return all_failures

    def _assert_worker_can_restart(self, worker_id: int, exception: Exception) -> None:
        """
        Checks if we can recover from an exception from a worker.
        If the restart limit is exceeded it will raise a UnityCommunicationException.
        If the exception is not recoverable it re-raises the exception.
        """
        if (
            isinstance(exception, UnityCommunicationException)
            or isinstance(exception, UnityTimeOutException)
            or isinstance(exception, UnityEnvironmentException)
            or isinstance(exception, UnityCommunicatorStoppedException)
        ):
            if self._worker_has_restart_quota(worker_id):
                return
            else:
                logger.error(
                    f"Worker {worker_id} exceeded the allowed number of restarts."
                )
                raise exception
        raise exception

    def _worker_has_restart_quota(self, worker_id: int) -> bool:
        self._drop_old_restart_timestamps(worker_id)
        max_lifetime_restarts = self.run_options.env_settings.max_lifetime_restarts
        max_limit_check = (
            max_lifetime_restarts == -1
            or self.restart_counts[worker_id] < max_lifetime_restarts
        )

        rate_limit_n = self.run_options.env_settings.restarts_rate_limit_n
        rate_limit_check = (
            rate_limit_n == -1
            or len(self.recent_restart_timestamps[worker_id]) < rate_limit_n
        )

        return rate_limit_check and max_limit_check

    def _drop_old_restart_timestamps(self, worker_id: int) -> None:
        """
        Drops environment restart timestamps that are outside of the current window.
        """

        def _filter(t: datetime.datetime) -> bool:
            return t > datetime.datetime.now() - datetime.timedelta(
                seconds=self.run_options.env_settings.restarts_rate_limit_period_s
            )

        self.recent_restart_timestamps[worker_id] = list(
            filter(_filter, self.recent_restart_timestamps[worker_id])
        )

    def _step(self) -> List[EnvironmentStep]:
        # Queue steps for any workers which aren't in the "waiting" state.
        self._queue_steps()

        worker_steps: List[EnvironmentResponse] = []
        step_workers: Set[int] = set()
        # Poll the step queue for completed steps from environment workers until we retrieve
        # 1 or more, which we will then return as StepInfos
        while len(worker_steps) < 1:
            # Attend que les worker aient step l'env => au moins une réponse ?
            try:
                while True:
                    # puis récupère les EnvironmentStep qui sont dans la réponse
                    # jusqu'à ce que step_queue soit vide
                    step: EnvironmentResponse = self.step_queue.get_nowait()
                    if step.cmd == EnvironmentCommand.ENV_EXITED:
                        # If even one env exits try to restart all envs that failed.
                        self._restart_failed_workers(step)
                        # Clear state and restart this function.
                        worker_steps.clear()
                        step_workers.clear()
                        self._queue_steps()
                    elif step.worker_id not in step_workers:
                        self.env_workers[step.worker_id].waiting = False
                        worker_steps.append(step)
                        step_workers.add(step.worker_id)
            except EmptyQueueException:
                pass
        step_infos = self._postprocess_steps(worker_steps)
        return step_infos

    def _reset_env(self) -> List[EnvironmentStep]:
        while any(ew.waiting for ew in self.env_workers):
            if not self.step_queue.empty():
                step = self.step_queue.get_nowait()
                self.env_workers[step.worker_id].waiting = False

        # First enqueue reset commands for all workers so that they reset in parallel
        for ew in self.env_workers:
            ew.send(EnvironmentCommand.RESET)
        # Next (synchronously) collect the reset observations from each worker in sequence
        for ew in self.env_workers:
            ew.previous_step = EnvironmentStep(ew.recv().payload, ew.worker_id, {}, {})
        return list(map(lambda ew: ew.previous_step, self.env_workers))


    @property
    def training_behaviors(self) -> BehaviorSpec:
        self.env_workers[0].send(EnvironmentCommand.BEHAVIOR_SPECS)
        result = list(self.env_workers[0].recv().payload._dict.values())[0]
        return result

    def close(self) -> None:
        logger.debug("SubprocessEnvManager closing.")
        for env_worker in self.env_workers:
            env_worker.request_close()
        # Pull messages out of the queue until every worker has CLOSED or we time out.
        deadline = time.time() + WORKER_SHUTDOWN_TIMEOUT_S
        while self.workers_alive > 0 and time.time() < deadline:
            try:
                step: EnvironmentResponse = self.step_queue.get_nowait()
                env_worker = self.env_workers[step.worker_id]
                if step.cmd == EnvironmentCommand.CLOSED and not env_worker.closed:
                    env_worker.closed = True
                    self.workers_alive -= 1
                # Discard all other messages.
            except EmptyQueueException:
                pass
        self.step_queue.close()
        # Sanity check to kill zombie workers and report an issue if they occur.
        if self.workers_alive > 0:
            logger.error("SubprocessEnvManager had workers that didn't signal shutdown")
            for env_worker in self.env_workers:
                if not env_worker.closed and env_worker.process.is_alive():
                    env_worker.process.terminate()
                    logger.error(
                        "A SubprocessEnvManager worker did not shut down correctly so it was forcefully terminated."
                    )
        self.step_queue.join_thread()

    def _postprocess_steps(self, env_steps: List[EnvironmentResponse]) -> List[EnvironmentStep]:
        # Converti les réponses en EnvironmentStep
        step_infos = []
        timer_nodes = []
        for step in env_steps:
            payload: StepResponse = step.payload
            env_worker = self.env_workers[step.worker_id]
            new_step = EnvironmentStep(
                payload.all_step_result,
                step.worker_id,
                env_worker.previous_all_action_info,
                payload.environment_stats,
            )
            step_infos.append(new_step)
            env_worker.previous_step = new_step

            if payload.timer_root:
                timer_nodes.append(payload.timer_root)

        if timer_nodes:
            with hierarchical_timer("workers") as main_timer_node:
                for worker_timer_node in timer_nodes:
                    main_timer_node.merge(
                        worker_timer_node, root_name="worker_root", is_parallel=True
                    )

        return step_infos

    @timed
    def _take_step(self, last_step: EnvironmentStep) -> ActionInfo:
        decision_step,_ = last_step.current_all_step_result
        return self.policy.get_action(decision_step, last_step.worker_id)

