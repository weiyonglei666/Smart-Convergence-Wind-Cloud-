import random
from copy import deepcopy
from typing import List, Optional

from langchain_core.documents import Document

from .planning.planner import Plan
from .prompts.relations import INTER_RELATIONS
from .rhetorical_graph.graph import RhetoricalGraph
from .rst_tree.tree import RstTree


def _select(n: int, ratio: float, rng: random.Random) -> List[int]:
    k = int(round(n * ratio))
    return rng.sample(range(n), k) if 0 < k <= n else []


def inject_noise(retrieved: List[Document], pool: List[Document], ratio: float, seed: int = 42) -> List[Document]:
    rng = random.Random(seed)
    retrieved_ids = {c.metadata["chunk_id"] for c in retrieved}
    distractors = [c for c in pool if c.metadata["chunk_id"] not in retrieved_ids]
    if not distractors:
        return list(retrieved)
    out = list(retrieved)
    positions = _select(len(out), ratio, rng)
    picked = rng.sample(distractors, min(len(positions), len(distractors)))
    for idx, chunk in zip(positions, picked):
        out[idx] = chunk
    return out


def perturb_rst(trees: List[RstTree], mode: str, ratio: float, seed: int = 42) -> List[RstTree]:
    rng = random.Random(seed)
    trees = deepcopy(trees)
    for tree in trees:
        if mode == "shuffle_relations":
            idx = _select(len(tree.relations), ratio, rng)
            labels = [tree.relations[i].relation for i in idx]
            rng.shuffle(labels)
            for i, label in zip(idx, labels):
                tree.relations[i].relation = label
        elif mode == "alter_nuclearity":
            for i in _select(len(tree.edus), ratio, rng):
                tree.edus[i].role = "S" if tree.edus[i].role == "N" else "N"
        elif mode == "remove_subtree":
            drop = set(_select(len(tree.relations), ratio, rng))
            tree.relations = [r for i, r in enumerate(tree.relations) if i not in drop]
    return trees


def perturb_graph(graph: RhetoricalGraph, mode: str, ratio: float, seed: int = 42) -> RhetoricalGraph:
    rng = random.Random(seed)
    graph = deepcopy(graph)
    edges = graph.edges
    related = [i for i, e in enumerate(edges) if e.relation != "UNRELATED"]
    count = int(round(len(related) * ratio))
    picked = rng.sample(related, count) if 0 < count <= len(related) else []
    if mode == "remove_edges":
        for i in picked:
            edges[i].relation = "UNRELATED"
    elif mode == "flip_directions":
        for i in picked:
            edges[i].source, edges[i].target = edges[i].target, edges[i].source
    elif mode == "replace_relations":
        for i in picked:
            options = [r for r in INTER_RELATIONS if r not in ("UNRELATED", edges[i].relation)]
            edges[i].relation = rng.choice(options)
    return graph


def perturb_plan(plan: Plan, mode: str, ratio: float, seed: int = 42) -> Optional[Plan]:
    rng = random.Random(seed)
    if mode == "remove_plan":
        return None
    plan = deepcopy(plan)
    steps = plan.steps
    if mode == "shuffle_steps":
        idx = _select(len(steps), ratio, rng)
        picked = [steps[i] for i in idx]
        rng.shuffle(picked)
        for i, j in zip(idx, range(len(picked))):
            steps[i] = picked[j]
    elif mode == "drop_steps":
        drop = set(_select(len(steps), ratio, rng))
        plan.steps = [s for i, s in enumerate(steps) if i not in drop]
    return plan
