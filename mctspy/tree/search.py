# search.py
import multiprocessing as mp
import time
from typing import Optional, List, Any, Tuple

from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.games.common import TwoPlayersAbstractGameState

from mctspy.utils import _pickle_method, _unpickle_method  # Import pickling utilities


def simulate_rollout(state: TwoPlayersAbstractGameState) -> Any:
    """
    Perform a rollout (simulation) from the given state.

    Parameters
    ----------
    state : TwoPlayersAbstractGameState
        The current game state.

    Returns
    -------
    Any
        The game result after the rollout.
    """
    current_rollout_state = state
    while not current_rollout_state.is_game_over():
        possible_moves = current_rollout_state.get_legal_actions()
        if not possible_moves:
            break  # No possible moves
        # Since 'rollout_policy' is a static method, access via class
        action = TwoPlayersGameMonteCarloTreeSearchNode.rollout_policy(possible_moves)
        current_rollout_state = current_rollout_state.move(action)
    return current_rollout_state.game_result


class MonteCarloTreeSearch:
    def __init__(self, root: TwoPlayersGameMonteCarloTreeSearchNode, num_processes: int = mp.cpu_count()):
        """
        MonteCarloTreeSearchNode

        Parameters
        ----------
        root : TwoPlayersGameMonteCarloTreeSearchNode
            The root node of the MCTS tree.
        num_processes : int
            Number of processes to use in parallel.
        """
        self.root = root
        self.num_processes = num_processes

    def best_action(
        self, simulations_number: Optional[int] = None, total_simulation_seconds: Optional[float] = None
    ) -> TwoPlayersGameMonteCarloTreeSearchNode:
        """
        Perform MCTS simulations and return the best action.

        Parameters
        ----------
        simulations_number : int, optional
            Number of simulations to perform.
        total_simulation_seconds : float, optional
            Maximum time to spend on simulations (in seconds).

        Returns
        -------
        TwoPlayersGameMonteCarloTreeSearchNode
            The child node with the highest visit count.
        """
        if simulations_number is None and total_simulation_seconds is None:
            raise ValueError("Either simulations_number or total_simulation_seconds must be specified")

        if simulations_number is not None:
            self._best_action_serial(simulations_number)
        else:
            self._best_action_time_limited(total_simulation_seconds)

        return self.root.best_child(c_param=0.0)

    def best_action_parallel(
        self, simulations_number: Optional[int] = None, total_simulation_seconds: Optional[float] = None
    ) -> TwoPlayersGameMonteCarloTreeSearchNode:
        """
        Perform MCTS simulations in parallel and return the best action.

        Parameters
        ----------
        simulations_number : int, optional
            Number of simulations to perform.
        total_simulation_seconds : float, optional
            Maximum time to spend on simulations (in seconds).

        Returns
        -------
        TwoPlayersGameMonteCarloTreeSearchNode
            The child node with the highest visit count.
        """
        if simulations_number is None and total_simulation_seconds is None:
            raise ValueError("Either simulations_number or total_simulation_seconds must be specified")

        with mp.Pool(processes=self.num_processes) as pool:
            if simulations_number is not None:
                self._best_action_parallel_simulations(pool, simulations_number)
            else:
                self._best_action_parallel_time(pool, total_simulation_seconds)

        return self.root.best_child(c_param=0.0)

    def _best_action_serial(self, simulations_number: int) -> None:
        """
        Perform simulations serially.

        Parameters
        ----------
        simulations_number : int
            Number of simulations to perform.
        """
        for _ in range(simulations_number):
            node = self._tree_policy()
            reward = node.rollout()
            node.backpropagate(reward)

    def _best_action_time_limited(self, total_simulation_seconds: float) -> None:
        """
        Perform simulations until the time limit is reached.

        Parameters
        ----------
        total_simulation_seconds : float
            Maximum time to spend on simulations (in seconds).
        """
        end_time = time.time() + total_simulation_seconds
        while time.time() < end_time:
            node = self._tree_policy()
            reward = node.rollout()
            node.backpropagate(reward)

    def _best_action_parallel_simulations(self, pool: mp.Pool, simulations_number: int) -> None:
        """
        Perform simulations in parallel based on the number of simulations.

        Parameters
        ----------
        pool : mp.Pool
            The multiprocessing pool.
        simulations_number : int
            Number of simulations to perform.
        """
        batch_size = self.num_processes * 10  # Adjust as needed for better performance
        simulations_done = 0

        while simulations_done < simulations_number:
            current_batch_size = min(batch_size, simulations_number - simulations_done)
            # Prepare the states for rollout
            nodes = [self._tree_policy() for _ in range(current_batch_size)]
            states = [node.state for node in nodes]

            # Perform parallel rollouts
            rewards = pool.map(simulate_rollout, states)

            # Backpropagate the results
            for node, reward in zip(nodes, rewards):
                node.backpropagate(reward)

            simulations_done += current_batch_size

    def _best_action_parallel_time(self, pool: mp.Pool, total_simulation_seconds: float) -> None:
        """
        Perform simulations in parallel based on a time limit.

        Parameters
        ----------
        pool : mp.Pool
            The multiprocessing pool.
        total_simulation_seconds : float
            Maximum time to spend on simulations (in seconds).
        """
        end_time = time.time() + total_simulation_seconds
        batch_size = self.num_processes * 10  # Adjust as needed for better performance

        while time.time() < end_time:
            remaining_time = end_time - time.time()
            if remaining_time <= 0:
                break

            # Prepare the states for rollout
            nodes = [self._tree_policy() for _ in range(batch_size)]
            states = [node.state for node in nodes]

            # Perform parallel rollouts with a timeout
            try:
                rewards = pool.map(simulate_rollout, states, timeout=remaining_time)
            except mp.TimeoutError:
                # If timeout occurs, get whatever is available
                rewards = pool.map(simulate_rollout, states)

            # Backpropagate the results
            for node, reward in zip(nodes, rewards):
                node.backpropagate(reward)

    def _tree_policy(self) -> TwoPlayersGameMonteCarloTreeSearchNode:
        """
        Select a node to run a simulation/playout for.

        Returns
        -------
        TwoPlayersGameMonteCarloTreeSearchNode
            The selected node.
        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
