"""Utility functions for predicate learning."""

import torch
from relational_structs import LiftedAtom

from skill_refactor.utils.structs import LiftedOperator, Predicate


def two2one(two: torch.Tensor) -> torch.Tensor:
    """Convert a 2D ae vector to a 1D tensor."""
    assert two.shape[-1] == 2
    assert torch.all(two >= 0)

    one = two[..., 0] - two[..., 1]
    assert torch.all(one <= 1), "Values should be in {0, 1}"
    return one


def one2two(one: torch.Tensor) -> torch.Tensor:
    """Convert a 1D ae vector to a 2D ae tensor."""
    assert one.shape[-1] == 1
    cate_vector_sampled = torch.tensor(one).unsqueeze(0)
    # from cate to n_row x n_channel
    vector_sampled = torch.zeros((1, one.shape[0], 2), dtype=torch.int)
    possible_values = torch.tensor([[0, 0], [1, 0], [0, 1]])
    vector_sampled = possible_values[cate_vector_sampled]
    two = vector_sampled.squeeze(0)
    assert two.shape[-1] == 2
    assert torch.all(two >= 0)
    return two


def operator_to_ae_vector(
    operator_set: set[LiftedOperator], name_list: list[str]
) -> dict[Predicate, dict[str, torch.Tensor]]:
    """Convert a set of operators to AE vector / Variable binding per predicate.

    Args:
        operator_set: Set of lifted operators
        name_list: Ordered list of operator names

    Returns:
        Dictionary mapping predicates to their AE vectors and variable bindings
    """

    # Step 1: Find all unique predicate instances (predicate + variable binding combinations)
    predicate_instances = {}  # Maps (predicate, var_bind_tuple) -> unique_predicate
    predicate_counter = {}  # Maps base predicate -> count for naming

    # Create name to operator mapping
    name_to_operator = {op.name: op for op in operator_set}

    # First pass: discover all unique predicate instances
    for op in operator_set:
        # Process add effects
        for effect_atom in op.add_effects:
            predicate = effect_atom.predicate
            var_bind_idx = _compute_variable_binding(effect_atom, op)
            var_bind_tuple = tuple(var_bind_idx.tolist())

            key = (predicate, var_bind_tuple)
            if key not in predicate_instances:
                # Create a new predicate instance
                if predicate not in predicate_counter:
                    predicate_counter[predicate] = 0

                if predicate_counter[predicate] == 0:
                    # First instance keeps original name
                    new_predicate = predicate  # type: ignore[assignment]
                else:
                    # Subsequent instances get numbered names
                    new_name = f"{predicate.name}{predicate_counter[predicate] + 1}"
                    new_predicate = Predicate(new_name, predicate.types)  # type: ignore[assignment]

                predicate_instances[key] = new_predicate
                predicate_counter[predicate] += 1

        # Process delete effects
        for effect_atom in op.delete_effects:
            predicate = effect_atom.predicate
            var_bind_idx = _compute_variable_binding(effect_atom, op)
            var_bind_tuple = tuple(var_bind_idx.tolist())

            key = (predicate, var_bind_tuple)
            if key not in predicate_instances:
                # Create a new predicate instance
                if predicate not in predicate_counter:
                    predicate_counter[predicate] = 0

                if predicate_counter[predicate] == 0:
                    # First instance keeps original name
                    new_predicate = predicate  # type: ignore[assignment]
                else:
                    # Subsequent instances get numbered names
                    new_name = f"{predicate.name}{predicate_counter[predicate] + 1}"
                    new_predicate = Predicate(new_name, predicate.types)  # type: ignore[assignment]

                predicate_instances[key] = new_predicate
                predicate_counter[predicate] += 1

        # NOTE: Do not process preconditions, only considering effects for AE vectors

    # Step 2: Initialize result dictionary
    num_operators = len(name_list)
    result = {}

    for (_, var_bind_tuple), unique_predicate in predicate_instances.items():
        result[unique_predicate] = {
            "ae_vector": torch.zeros(num_operators, 2),  # [add_effect, delete_effect]
            "var_bind_idx": torch.tensor(var_bind_tuple, dtype=torch.long),
        }

    # Step 3: Process each operator to fill AE vectors
    for op_idx, op_name in enumerate(name_list):
        if op_name not in name_to_operator:
            continue

        operator = name_to_operator[op_name]

        # Process add effects
        for effect_atom in operator.add_effects:
            predicate = effect_atom.predicate
            var_bind_idx = _compute_variable_binding(effect_atom, operator)
            var_bind_tuple = tuple(var_bind_idx.tolist())

            key = (predicate, var_bind_tuple)
            if key in predicate_instances:
                unique_predicate = predicate_instances[key]
                result[unique_predicate]["ae_vector"][op_idx, 0] = 1.0

        # Process delete effects
        for effect_atom in operator.delete_effects:
            predicate = effect_atom.predicate
            var_bind_idx = _compute_variable_binding(effect_atom, operator)
            var_bind_tuple = tuple(var_bind_idx.tolist())

            key = (predicate, var_bind_tuple)
            if key in predicate_instances:
                unique_predicate = predicate_instances[key]
                result[unique_predicate]["ae_vector"][op_idx, 1] = 1.0

    return result  # type: ignore[return-value]


def _compute_variable_binding(
    atom: LiftedAtom, operator: LiftedOperator
) -> torch.Tensor:
    """Compute the variable binding indices for a given atom based on the operator."""
    var_bind_idx = torch.zeros(atom.predicate.arity, dtype=torch.long)

    for arg_idx, arg in enumerate(atom.variables):
        # Find which parameter this argument corresponds to
        param_idx = None
        for i, param in enumerate(operator.parameters):
            if param == arg:
                param_idx = i
                break

        if param_idx is not None:
            # Find the index among parameters of the same type
            param_type = atom.predicate.types[arg_idx]
            type_index = 0
            for i in range(param_idx):
                if operator.parameters[i].type == param_type:
                    type_index += 1
            var_bind_idx[arg_idx] = type_index

    return var_bind_idx
