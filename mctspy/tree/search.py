# search.py
import copyreg
import multiprocessing as mp
import time
from typing import Optional, List, Any, Tuple
import types

from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.utils import _pickle_method, _unpickle_method


class MonteCarloTreeSearch:

    def __init__(self, node: MonteCarloTreeSearchNode, num_processes: int = mp.cpu_count()):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        num_processes : int
            Number of processes to use in parallel
        """
        self.root = node
        self.num_processes = num_processes

    def best_action(self, simulations_number: int = None, total_simulation_seconds: float = None):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        total_simulation_seconds : float
            Amount of time the algorithm has to run. Specified in seconds

        Returns
        -------

        """

        if simulations_number is None:
            assert total_simulation_seconds is not None
            end_time = time.time() + total_simulation_seconds
            while True:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else:
            for _ in range(0, simulations_number):
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.0)

    def best_action_parallel(
        self, simulations_number: Optional[int] = None, total_simulation_seconds: Optional[float] = None
    ):
        if simulations_number is None and total_simulation_seconds is None:
            raise ValueError("Either simulations_number or total_simulation_seconds must be specified")

        with mp.Pool(processes=self.num_processes) as pool:
            if total_simulation_seconds is not None:
                end_time = time.time() + total_simulation_seconds
                while time.time() < end_time:
                    results = pool.map(self._parallel_simulate, [self.root] * self.num_processes)
                    for path, reward in results:
                        self._update_stats(path, reward)
            else:
                simulations_per_process = max(1, simulations_number // self.num_processes)
                remaining_simulations = simulations_number % self.num_processes

                results = pool.map(self._parallel_simulate,
                                   [self.root] * (self.num_processes * simulations_per_process + remaining_simulations))

                for path, reward in results:
                    self._update_stats(path, reward)

        return self.root.best_child(c_param=0.0)

    @staticmethod
    def _parallel_simulate(node) -> Tuple[List[Any], Any]:
        current_node = node
        path = [current_node]
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                current_node = current_node.expand()
                path.append(current_node)
                break
            else:
                current_node = current_node.best_child()
                path.append(current_node)

        reward = current_node.rollout()
        return path, reward

    def _update_stats(self, path: List[MonteCarloTreeSearchNode], reward: Any):
        for node in reversed(path):
            node.backpropagate(reward)

    def _tree_policy(self):
        """
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
