"""Terminal predicate learning for DSG baseline.

Learns neural classifiers that detect when skills successfully terminate (reach their
effect sets).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from relational_structs import Object
from torch import Tensor

from skill_refactor.approaches.pred_learner.base import BasePredicateLearner
from skill_refactor.approaches.pred_learner.neural_dataset import (
    OperatorTransition,
    create_train_val_dataloaders,
    generate_in_out_predicate_data_terminal,
    sample_middle_state_graphs,
    train_predicate_model,
)
from skill_refactor.approaches.pred_learner.neural_models import (
    create_neural_predicate_interpreter,
    setup_predicate_net,
    setup_predicate_optimizer,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import LiftedOperator, Predicate


class TerminalPredicateLearner(BasePredicateLearner):
    """Learn terminal predicates for each skill in DSG baseline.

    Terminal predicates classify whether a skill has successfully reached its goal
    state. These are used as effect sets in the DSG skill graph.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.action_names: List[str] = []
        self.operators: List[LiftedOperator] = []
        self._extract_action_names()
        self.action_to_index = {name: i for i, name in enumerate(self.action_names)}

        if self._verbose:
            logging.info(
                f"Initialized TerminalPredicateLearner with {len(self.action_names)} skills"
            )
            logging.info(f"Skill names: {self.action_names}")

    def _extract_action_names(self) -> None:
        action_names_set = set()
        lifted_operator_set = set()
        for traj in self._trajectories:
            for action_result in traj.actions:
                if action_result.has_op():
                    ground_op = action_result.get_op()
                    action_names_set.add(ground_op.parent.name)
                    lifted_operator_set.add(ground_op.parent)

        self.action_names = sorted(list(action_names_set))
        self.operators = sorted(list(lifted_operator_set), key=lambda x: x.name)

        for i, op in enumerate(self.operators):
            if op.name != self.action_names[i]:
                raise ValueError(
                    f"Action name mismatch: {op.name} != {self.action_names[i]}"
                )

    def learn_terminal_predicates_for_all_skills(
        self,
        predicate_configs: Dict[str, Any],
        save_dir: Path | None = None,
    ) -> Dict[
        str, Tuple[Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]]
    ]:
        """Learn terminal predicates for all skills.

        Args:
            predicate_configs: Dict mapping predicate names to their configs
            save_dir: Optional directory to save trained models

        Returns:
            Dict mapping skill names to (terminal_predicate, interpreter_function)
        """
        terminal_predicates = {}

        for skill_name in self.action_names:
            pred_name = f"terminal_{skill_name}"
            if pred_name not in predicate_configs:
                logging.warning(f"No config found for {pred_name}, skipping")
                continue

            logging.info(f"Learning terminal predicate for skill: {skill_name}")
            terminal_pred, interpreter = self.learn_terminal_predicate(
                skill_name=skill_name,
                predicate_config=predicate_configs[pred_name],
                save_dir=save_dir,
            )
            terminal_predicates[skill_name] = (terminal_pred, interpreter)

        return terminal_predicates

    def learn_terminal_predicate(
        self,
        skill_name: str,  # pylint: disable=unused-argument
        predicate_config: Dict[str, Any],
        save_dir: Path | None = None,
    ) -> Tuple[Predicate, Callable[[Tensor, List[Sequence[Object]]], Tensor]]:
        """Learn a terminal predicate for a specific skill."""
        pred_name = predicate_config["name"]
        pred_types = [
            self._tamp_system.components.type_container[t]
            for t in predicate_config["types"]
        ]

        terminal_pred = Predicate(pred_name, pred_types)

        ae_config = predicate_config["ae_generator"]
        assert (
            ae_config["name"] == "fixed"
        ), "Terminal predicates must use fixed AE generator"
        ae_vectors = torch.tensor(ae_config["kwargs"]["ae_vectors"])
        # ae_var_ids = ae_config["kwargs"]["ae_var_ids"]

        assert (
            ae_vectors.shape[0] == 1
        ), "Terminal predicates should have exactly one AE vector"
        ae_vector = ae_vectors[0]
        # var_bind_idx = torch.tensor(ae_var_ids[0])

        operator_transitions = self._prepare_operator_transitions(
            max_num_middle_states=5
        )

        (
            input_data_list,
            input_data_list_,
            input_middle_data_list,
            target_data_list,
            target_data_list_,
        ) = generate_in_out_predicate_data_terminal(
            operator_transitions,
            terminal_pred,
            ae_vector,
            self.action_to_index,
        )

        if len(input_data_list) == 0:
            raise ValueError(f"No training data generated for {pred_name}")

        train_loader, val_loader = create_train_val_dataloaders(
            input_data_list,
            target_data_list,
            input_data_list_,
            target_data_list_,
            input_middle_data_list,
            train_ratio=0.8,
            batch_size=predicate_config.get("batch_size", 64),
            shuffle_train=True,
        )

        nn_config = predicate_config["nn_archi"]
        predicate_net = setup_predicate_net(archi=nn_config)

        optimizer_config = predicate_config["optimizer"]
        lr_scheduler_config = predicate_config.get("lr_scheduler", None)

        optimizer, scheduler = setup_predicate_optimizer(
            predicate_net,
            opti_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_weights, _, _ = train_predicate_model(
            predicate_net,
            train_loader,
            val_loader,
            optimizer,
            super_label=CFG.super_label,
            num_epochs=predicate_config.get("num_epochs", 100),
            device=device,
            scheduler=scheduler,
            val_freq=predicate_config.get("val_freq", 10),
        )

        predicate_net.load_state_dict(best_weights)

        if save_dir is not None:
            save_path = save_dir / f"{pred_name}.pt"
            torch.save(best_weights, save_path)
            logging.info(f"Saved terminal predicate model to {save_path}")

        interpreter = create_neural_predicate_interpreter(
            predicate_net,
            terminal_pred,
            self._tamp_system,
            predicate_config.get("cls_threshold", 0.5),
            device,
        )

        return terminal_pred, interpreter

    def _prepare_operator_transitions(
        self, max_num_middle_states: int = 5
    ) -> List[OperatorTransition]:
        """Prepare operator transition data from trajectories."""
        from skill_refactor.approaches.operator_learner.segmentation import (
            segment_trajectory,
        )

        operator_transitions = []

        for traj, atom_seq in self._ground_atom_dataset:
            segments = segment_trajectory(traj, atom_seq)

            for segment in segments:
                pre_graph = self._tamp_system.state_to_graph(
                    segment.states[0].unsqueeze(0)
                )[0]
                post_graph = self._tamp_system.state_to_graph(
                    segment.states[-1].unsqueeze(0)
                )[0]

                middle_graphs = sample_middle_state_graphs(
                    segment,
                    self._tamp_system,
                    max_num_middle_states,
                )

                operator_transitions.append(
                    OperatorTransition(
                        pre_state_graph=pre_graph,
                        post_state_graph=post_graph,
                        pre_atoms=segment.init_atoms,
                        post_atoms=segment.final_atoms,
                        operator=segment.op,
                        middle_state_graphs=middle_graphs,
                    )
                )

        return operator_transitions
