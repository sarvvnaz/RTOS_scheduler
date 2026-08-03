"""Mapping nodes onto cores (OC-HEFT).

Step 1 of the algorithm, and the only part in this file so far, is the
*upward rank* from HEFT. It decides the ORDER in which nodes are placed,
not where they go. Choosing the core comes later.

    Rank(v) = c_v + max over successors s of ( comm(v, s) + Rank(s) )

Read it as: "how much work is still ahead of me before this task graph can
finish, counting myself?" A node with a large rank sits at the head of a
long remaining chain, so it is placed first and gets the best core.

The sink has no successors, so its rank is just its own execution time
(zero, because the sink is a dummy node). Every other rank builds on the
ranks behind it, which is why the nodes are walked in REVERSE topological
order: by the time we reach a node, all of its successors already have a
rank.
"""

from typing import Dict, List

import networkx as nx

from model import Task


def upward_rank(task: Task) -> Dict[int, float]:
    """Upward rank of every node, keyed by node id.

    Walks the graph backwards, from the sink towards the source.
    """
    rank: Dict[int, float] = {}

    for node_id in reversed(list(nx.topological_sort(task.graph))):
        successors = list(task.graph.successors(node_id))

        if successors:
            # the longest way onwards, through whichever successor is worst
            ahead = max(task.comm_cost(node_id, s) + rank[s]
                        for s in successors)
        else:
            ahead = 0.0                      # the sink: nothing comes after

        rank[node_id] = task.nodes[node_id].wcet + ahead

    return rank


def placement_order(task: Task) -> List[int]:
    """Node ids sorted by decreasing rank: the order to place them in.

    The highest rank goes first, because that node begins the longest
    remaining chain of work and so constrains the schedule the most.
    """
    rank = upward_rank(task)
    return sorted(rank, key=lambda node_id: -rank[node_id])


def rank_path(task: Task) -> List[int]:
    """The chain the ranking considers most urgent.

    Starts at the source and always follows the successor with the highest
    rank. This is the path whose length the source's rank measures.
    """
    rank = upward_rank(task)
    path = [task.source_id]

    while True:
        successors = list(task.graph.successors(path[-1]))
        if not successors:
            return path
        path.append(max(successors, key=lambda s: rank[s]))
