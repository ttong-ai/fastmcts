# search.py
import copyreg
import multiprocessing as mp
import time
from typing import Optional, List, Any, Tuple
import types

from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.utils import _pickle_method, _unpickle_method


class MonteCarloTreeSearch:
    def __init__(self, root: MonteCarloTreeSearchNode, num_processes: int = mp.cpu_count()):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        root : mctspy.tree.nodes.MonteCarloTreeSearchNode
        num_processes : int
            Number of processes to use in parallel
        """
        self.root = root
        self.num_processes = num_processes

    def best_action(
        self, simulations_number: Optional[int] = None, total_simulation_seconds: Optional[float] = None
    ) -> MonteCarloTreeSearchNode:
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
        MonteCarloTreeSearchNode
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
    ) -> MonteCarloTreeSearchNode:
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
        MonteCarloTreeSearchNode
            The child node with the highest visit count.
        """
        if simulations_number is None and total_simulation_seconds is None:
            raise ValueError("Either simulations_number or total_simulation_seconds must be specified")

        with mp.Pool(processes=self.num_processes) as pool:
            if total_simulation_seconds is not None:
                end_time = time.time() + total_simulation_seconds
                while time.time() < end_time:
                    # Prepare a batch of simulations
                    batch_size = self.num_processes
                    tasks = pool.map_async(self._simulate_once, [self.root] * batch_size)
                    try:
                        results = tasks.get(timeout=end_time - time.time())
                    except mp.TimeoutError:
                        results = tasks.get()
                    for reward in results:
                        self.root.backpropagate(reward)
            else:
                # Distribute simulations across processes
                simulations_per_process = simulations_number // self.num_processes
                remaining_simulations = simulations_number % self.num_processes

                total_tasks = [self.root] * (self.num_processes * simulations_per_process + remaining_simulations)
                rewards = pool.map(self._simulate_once, total_tasks)

                for reward in rewards:
                    self.root.backpropagate(reward)

        return self.root.best_child(c_param=0.0)

    def _simulate_once(self, node: MonteCarloTreeSearchNode) -> Any:
        """
        Perform a single MCTS simulation (selection, expansion, simulation, backpropagation).

        Parameters
        ----------
        node : MonteCarloTreeSearchNode
            The root node for simulation.

        Returns
        -------
        Any
            The simulation result to backpropagate.
        """
        path = []
        current_node = node
        # Selection and Expansion
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                current_node = current_node.expand()
                path.append(current_node)
                break
            else:
                current_node = current_node.best_child()
                path.append(current_node)
        # Simulation
        reward = current_node.rollout()
        return reward

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

    def _tree_policy(self) -> MonteCarloTreeSearchNode:
        """
        Select a node to run a simulation/playout for.

        Returns
        -------
        MonteCarloTreeSearchNode
            The selected node.
        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
