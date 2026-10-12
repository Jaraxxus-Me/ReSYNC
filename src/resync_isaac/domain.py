"""Strict inference-only reconstruction of a saved neural/symbolic domain.

Uses the same network and quantifier interpreters as training, without invoking
predicate invention or needing demonstration trajectories at inference time.
"""

import json
from pathlib import Path

import torch
import yaml
from relational_structs import LiftedAtom, Variable

from skill_refactor.approaches.pred_learner.neural_models import (
    create_neural_predicate_interpreter,
    create_quantified_predicate,
)
from skill_refactor.utils.structs import Predicate, LiftedOperator
from resync_isaac.neural_models import setup_predicate_net


def load_domain(system, directory: Path, config: Path):
    data = json.loads(
        (directory / "invented_predicates_operators_sc1.json").read_text()
    )
    candidates = dict(system.perceiver.predicate_interpreters)
    types = system.components.type_container
    for spec in yaml.safe_load(config.read_text())["predicates"]:
        for model_path in sorted(directory.glob(spec["name"] + "_*_model.pth")):
            name = model_path.name.removesuffix("_model.pth")
            model = setup_predicate_net(spec["nn_archi"])
            model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True),
                strict=True,
            )
            pred = Predicate(name, [types[t] for t in spec["types"]])
            candidates[pred] = create_neural_predicate_interpreter(
                model,
                pred,
                system,
                spec.get("cls_threshold", 0.5),
                torch.device(system.env.unwrapped.device),
            )
    for pred, interp in list(candidates.items()):
        for quantifier in ("", "Exist", "ForAll"):
            for variable in range(pred.arity):
                for negation in (False, True):
                    if quantifier == "" and (not negation or variable > 0):
                        continue
                    qp, qi = create_quantified_predicate(
                        pred, interp, system, quantifier, variable, negation
                    )
                    candidates[qp] = qi
    by_pddl = {p.pddl_str: (p, fn) for p, fn in candidates.items()}
    required = set(data["invented_predicates"])
    for op in data["operators"]:
        for key in ("preconditions", "add_effects", "delete_effects"):
            required.update(op[key])
    missing = required - by_pddl.keys()
    if missing:
        raise ValueError(
            f"Saved domain references missing predicates: {sorted(missing)}"
        )
    for key in sorted(required):
        p, fn = by_pddl[key]
        if p.name not in system.perceiver.predicates_container.predicates:
            system.perceiver.add_predicate_interpreter(
                p.name, p.types, fn, p.base_types
            )
    operators = set()
    for op in data["operators"]:
        variables = [Variable(n, types[t]) for n, t in op["parameters"]]
        atoms = {}
        for key in ("preconditions", "add_effects", "delete_effects"):
            atoms[key] = {
                LiftedAtom(by_pddl[pddl][0], [variables[i] for i in binding])
                for pddl, bindings in op[key].items()
                for binding in bindings
            }
        operators.add(LiftedOperator(op["name"], variables, **atoms))
    return operators


def repair_flipped_composition(operators):
    """Preserve the source demo's explicit repairs for multi-block composition."""
    clear_name = "ForAll_robot_0_Not_b_sc1p1_0"
    all_up = "ForAll_block_0_u_sc1p1_0"
    clear = {
        a
        for op in operators
        for a in op.preconditions | op.add_effects | op.delete_effects
        if a.predicate.name == clear_name
    }
    return {
        op.copy_with(
            preconditions={a for a in op.preconditions if a.predicate.name != all_up},
            add_effects={a for a in op.add_effects if a.predicate.name != all_up}
            | (clear if op.name == "Flip" else set()),
            delete_effects=set(op.delete_effects),
        )
        for op in operators
    }
