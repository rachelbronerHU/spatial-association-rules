"""
FP-tree over weights: adds up float weights instead of counting rows.

Node weights are an upper bound on min-based support, so the tree only proposes
candidates and attraction.py measures each one exactly.
"""


class _Node:
    __slots__ = ("item", "weight", "parent", "children", "next_same_item")

    def __init__(self, item, parent):
        self.item = item
        self.weight = 0.0
        self.parent = parent
        self.children = {}
        self.next_same_item = None


class FPTree:
    """header: item -> [total weight, first node]. Nodes chain via next_same_item."""

    def __init__(self):
        self.root = _Node(None, None)
        self.header = {}

    def count_items(self, transactions):
        """First pass: total weight per item."""
        for transaction in transactions:
            for item, weight in transaction.items():
                entry = self.header.setdefault(item, [0.0, None])
                entry[0] += weight

    def drop_light_items(self, min_weight):
        """Remove items too light to carry an itemset."""
        self.header = {item: entry for item, entry in self.header.items()
                       if entry[0] >= min_weight}

    def add(self, transaction):
        """Insert one transaction, adding its weights along the path."""
        items = [item for item in transaction if item in self.header]
        items.sort(key=lambda item: self.header[item][0], reverse=True)

        node = self.root
        for item in items:
            child = node.children.get(item)
            if child is None:
                child = _Node(item, node)
                child.next_same_item = self.header[item][1]
                self.header[item][1] = child
                node.children[item] = child
            child.weight += transaction[item]
            node = child

    def paths_to(self, item):
        """The prefix paths leading to each node of an item, weighted by that node."""
        paths = []
        node = self.header[item][1]
        while node is not None:
            path = {}
            parent = node.parent
            while parent.item is not None:
                path[parent.item] = path.get(parent.item, 0.0) + node.weight
                parent = parent.parent
            if path:
                paths.append(path)
            node = node.next_same_item
        return paths


def find_itemsets(transactions, min_weight, max_items):
    """
    All candidate itemsets carrying at least min_weight, as frozensets.

    max_items stops the recursion rather than filtering afterwards.
    """
    tree = FPTree()
    tree.count_items(transactions)
    tree.drop_light_items(min_weight)
    for transaction in transactions:
        tree.add(transaction)
    return _grow(tree, min_weight, max_items, prefix=[])


def _grow(tree, min_weight, max_items, prefix):
    """Extend the prefix one item at a time, mining each conditional tree."""
    found = []

    for item in sorted(tree.header, key=lambda i: tree.header[i][0]):
        longer = prefix + [item]
        found.append(frozenset(longer))

        if len(longer) >= max_items:
            continue

        paths = tree.paths_to(item)
        if not paths:
            continue

        # support({A,B}) <= support({A}), so pruning here cannot hide a heavier set.
        conditional = FPTree()
        conditional.count_items(paths)
        conditional.drop_light_items(min_weight)
        for path in paths:
            conditional.add(path)

        found.extend(_grow(conditional, min_weight, max_items, longer))

    return found
