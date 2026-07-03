"""Reaction network graph data structures for synthesis pathways."""

import collections
import copy
import dataclasses
import numbers
import pickle
from collections.abc import Callable, Iterable
from typing import Literal, Self

import tqdm

# import line_profiler
from rdkit import Chem

from synlad.utils import synthesis_chem_utils as chem_utils
from synlad.utils.synthesis_utils import ManyToManyMapping, TwoWayDict

# Type alias for representing a reaction as (reactants, product) using SMILES strings
ReactionType = tuple[frozenset[str], str]


@dataclasses.dataclass
class MolecularNode:
    """
    Holds a molecular node in a reaction network graph.
    """

    canon_smi: str
    _metadata: dict = dataclasses.field(default_factory=dict)
    # ^ bonus data, e.g., molecular properties. Not used in class hash/equality.

    def post_init(self):
        if Chem.MolFromSmiles(self.canon_smi) is None:
            raise ValueError(f"Invalid SMILES: {self.canon_smi}")

    def __hash__(self):
        return hash(self.canon_smi)

    def __eq__(self, other):
        return self.canon_smi == other.canon_smi

    @classmethod
    def from_smiles(cls, smiles: str, _metadata):
        canonical_smiles = chem_utils.canonicalize(smiles)
        return cls(canonical_smiles, _metadata)

    def __repr__(self):
        return self.canon_smi


@dataclasses.dataclass
class ReactionEdge:
    """
    Holds a molecular reaction in a reaction network graph. Note currently assume only one product per reaction.
    Ignore reagents and physical conditions.
    """

    reactants: frozenset[MolecularNode]
    product: MolecularNode
    _metadata: dict = dataclasses.field(default_factory=dict)
    # ^ bonus data, e.g., example reactions from a reaction dataset. Not used in class hash/equality.

    def __repr__(self):
        reactants_prt = ".".join(
            sorted((o.canon_smi for o in self.reactants), key=lambda x: hash(x))
        )
        # ^ hash to ensure consistent order.
        return f"{reactants_prt}>>{self.product.canon_smi}"

    def __hash__(self):
        return hash((self.reactants, self.product))

    def __eq__(self, other: Self):
        return self.reactants == other.reactants and self.product == other.product

    @property
    def all_molecules(self):
        return (*self.reactants, self.product)


class ReactionNetwork:
    """
    Class to hold a reaction network graph. This is a directed hypergraph where nodes are molecules and edges are
    reactions.
    """

    def __init__(self):
        self._canon_smi_to_molecule: TwoWayDict[str, MolecularNode] = TwoWayDict()
        # ^ one to one mapping from canonical SMILES to MolecularNode objects.
        self._reactant_product_to_reaction: TwoWayDict[
            tuple[frozenset[MolecularNode], MolecularNode], ReactionEdge
        ] = TwoWayDict()
        # ^ one to one mapping from (reactants, product) to ReactionEdge objects.
        self._mol_to_reactions: ManyToManyMapping[MolecularNode, set[ReactionEdge]] = (
            ManyToManyMapping()
        )
        # ^ maps Molecule Nodes to the set of reactions that it is involved in.

    def add_reaction_from_rxnsmi(self, rxn_smi: str, metadata_updater: Callable | None = None):
        if metadata_updater is None:

            def metadata_updater(_):
                return None

        reactants, _, product = rxn_smi.split(">")
        # ^ we will ignore the reagents.

        if "." in reactants:
            reactants = reactants.split(".")
        else:
            reactants = [reactants]

        if "." in product:
            raise ValueError("Only one product per reaction supported currently.")

        rxn = self.add_reaction_smi_sets(reactants, product, metadata_updater)

        return rxn

    def __getstate__(self):
        """overriding default as may add more methods to this class in future."""
        return dict(
            _canon_smi_to_molecule=self._canon_smi_to_molecule.forward_mapping_only,
            _reactant_product_to_reaction=self._reactant_product_to_reaction.forward_mapping_only,
            _mol_to_reactions=self._mol_to_reactions.forward_mapping_only,
        )

    def __setstate__(self, state):
        """overriding default as may add more methods to this class in future."""
        self._canon_smi_to_molecule = TwoWayDict.from_forward_mapping(
            state["_canon_smi_to_molecule"]
        )
        self._reactant_product_to_reaction = TwoWayDict.from_forward_mapping(
            state["_reactant_product_to_reaction"]
        )
        self._mol_to_reactions = ManyToManyMapping.from_forward_mapping(state["_mol_to_reactions"])

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def add_reaction_smi_sets(
        self,
        reactant_smi_sets: Iterable[str],
        product_smi: str,
        metadata_updater,
        canonicalize=True,
    ):
        canon = chem_utils.canonicalize if canonicalize else lambda x: x
        reactants = [canon(smi) for smi in reactant_smi_sets]
        product = canon(product_smi)

        if any(o is None for o in reactants) or product is None:
            raise ValueError("Invalid SMILES in reaction.")

        reactants = [self._get_or_create_molecule(smi) for smi in reactants]
        product = self._get_or_create_molecule(product)

        rxn = self._get_or_create_reaction(frozenset(reactants), product)
        metadata_updater(rxn._metadata)

        for m in [*reactants, product]:
            self._mol_to_reactions.add_single_relationship(m, rxn)
        return rxn

    def add_smi_sets(self, smi_sets: Iterable[str], canonicalize=True):
        """Just adds the set of SMILES to the graph; returns the created molecular nodes."""
        if canonicalize:
            smi_sets = [chem_utils.canonicalize(smi) for smi in smi_sets]
        if any(o is None for o in smi_sets):
            raise ValueError(f"Invalid SMILES found in set: {smi_sets}")
        out = [self._get_or_create_molecule(smi) for smi in smi_sets]
        return out

    def _get_or_create_molecule(self, canon_smi: str):
        out = self._canon_smi_to_molecule.setdefault(canon_smi, MolecularNode(canon_smi))
        assert out.canon_smi == canon_smi, f"canonicalizing inconsistent for {canon_smi}"
        return out

    def _get_or_create_reaction(self, reactants: frozenset[MolecularNode], product: MolecularNode):
        out = self._reactant_product_to_reaction.setdefault(
            (reactants, product), ReactionEdge(reactants, product)
        )
        return out

    def prune_node(self, node: MolecularNode):
        """
        Removes a node from the graph and all reactions that involve it (either as a reactant or a product).

        Note that this will **not** remove dangling left over nodes.
        """
        canon_smi = node.canon_smi

        # remove node (will give key error if not found)
        node = self._canon_smi_to_molecule.pop(canon_smi)

        # work out what reactions that this node is involved in and remove that
        if node in self._mol_to_reactions:
            reactions = self._mol_to_reactions.pop(node)
            for rxn in reactions:
                rxn: ReactionEdge
                del self._reactant_product_to_reaction[(rxn.reactants, rxn.product)]
                other_mols = set(rxn.all_molecules) - {node}
                for m in other_mols:
                    self._mol_to_reactions.remove_single_relationship(m, rxn)

    def prune_reaction(self, rxn: ReactionEdge):
        """
        Removes a reaction from the graph. If the product of this reaction is no longer attached to the other nodes then
        will be removed. However, does not remove the reactants

        Note that this will **not** remove dangling left over nodes.
        """
        reactants = rxn.reactants
        product = rxn.product

        # remove reaction
        del self._reactant_product_to_reaction[(reactants, product)]

        # remove reaction from all molecules
        for m in self.get_inverse_relationships(rxn):
            self._mol_to_reactions.remove_single_relationship(m, rxn)

    def clean_dangling_nodes(self):
        """
        Removes all nodes that are not involved in any reactions.
        """
        for m in list(self.nodes):
            if m not in self._mol_to_reactions:
                self.prune_node(m)

    def subset_network_return_copy(
        self,
        mol_set: set[MolecularNode],
        kept_reactions: (
            set[ReactionEdge] | Callable[[set[ReactionEdge]], set[ReactionEdge]] | None
        ) = None,
    ):
        """Create a new `ReactionNetwork` object that is a subset of the current one. This is defined by a set of
        molecular nodes and (optionally) a set of reaction nodes.

        Note that the nodes and edges will be copies...

        Arguments:
            mol_set -- set of `MolecularNode` objects to include in the subset.

        Keyword Arguments:
            reactions -- this describes which `ReactionEdge` edges to keep in the subset. If None then will keep all
                between existing molecules. If a set of `ReactionEdge` objects then will keep just those. If a function
                then will call these on each set of `ReactionEdge`s leading from each product (default: {None}).

        """
        assert len(mol_set) > 0, "Must have at least one molecule in subset."

        # Get the nodes
        nodes_to_keep = [copy.copy(m) for m in mol_set]
        new_canon_canon_smi_to_molecule = TwoWayDict(**{m.canon_smi: m for m in nodes_to_keep})

        # Get the reactions that will be kept
        added_reactions = set()
        for m in mol_set:
            plausible_reactions_to_add = self._mol_to_reactions[m]
            for r in plausible_reactions_to_add:
                # if already added or not all the molecules are being kept, skip.
                if r in added_reactions or (not all(o in mol_set for o in r.all_molecules)):
                    continue

                # then deciding if we should keep this reaction.
                # - if no filter, keep.
                if kept_reactions is None:
                    added_reactions.add(r)
                # - if kept_reactions is a set and r is a member, keep.
                elif isinstance(kept_reactions, set):
                    if r in kept_reactions:
                        added_reactions.add(r)
                # - if kept_reactions is a function, call it on the set of reactions that belong to product and keep
                # if this r is in the set.
                elif callable(kept_reactions):
                    all_product_reactions = [
                        r_ for r_ in self._mol_to_reactions[r.product] if r_.product == r.product
                    ]
                    all_plausible_product_reactions = {
                        r_
                        for r_ in all_product_reactions
                        if all(m_ in mol_set for m_ in r_.all_molecules)
                    }
                    rxns_to_keep = kept_reactions(all_plausible_product_reactions)
                    added_reactions.update(rxns_to_keep)
                    # ^ note here we add all at once. we can't guarantee that the kept_reactions function will be
                    # deterministic.
                else:
                    raise NotImplementedError("kept_reactions must be None, a set, or a function.")

        # set up the remaining datastructures -- make copies of the reactions.
        new_reactant_product_to_reaction = TwoWayDict()
        new_mol_to_reactions = ManyToManyMapping()
        for r in added_reactions:
            new_reactants = frozenset(
                new_canon_canon_smi_to_molecule[m.canon_smi] for m in r.reactants
            )
            new_product = new_canon_canon_smi_to_molecule[r.product.canon_smi]
            new_r = ReactionEdge(new_reactants, new_product)
            new_reactant_product_to_reaction[(new_reactants, new_product)] = new_r
            for m in new_r.all_molecules:
                new_mol_to_reactions.add_single_relationship(m, new_r)

        # create the new network
        new_network = ReactionNetwork()
        new_network._canon_smi_to_molecule = new_canon_canon_smi_to_molecule
        new_network._reactant_product_to_reaction = new_reactant_product_to_reaction
        new_network._mol_to_reactions = new_mol_to_reactions

        return new_network

    def get_min_retrosyntheses(
        self,
        mol_to_retrosynthesize: MolecularNode,
        distances_to_building_blocks: dict[MolecularNode, int],
    ) -> Iterable["Self"]:
        """
        Returns a generator of miniumum retrosyntheses RxnNets from the given molecule to the building blocks.

        Returns generator as there may be several reaction networks with the same depth (does not pick between these).

        But at moment generator just yields one reaction network -- have not implemented the others yet!
        """
        mol_to_retrosynthesize_queue = collections.deque(
            [mol_to_retrosynthesize]
        )  # will use as a LIFO queue.
        seen_mols = set(mol_to_retrosynthesize_queue)
        seen_reactions = set()
        processed_mols = set()

        while len(mol_to_retrosynthesize_queue) > 0:
            next_mol_to_retrosynthesize = mol_to_retrosynthesize_queue.popleft()

            # v could be added to the queue several times, so skip if already seen and processed it:
            if next_mol_to_retrosynthesize in processed_mols:
                continue
            else:
                processed_mols.add(next_mol_to_retrosynthesize)

            # we want to assign a score to each reaction that produces the current molecule. this will be the height of
            # the highest reactant (i.e., the one that is furthest from a building block).
            reactions_to_scores = {}
            for rxn in self._mol_to_reactions[next_mol_to_retrosynthesize]:
                if rxn.product == next_mol_to_retrosynthesize:
                    reactions_to_scores[rxn] = max(
                        distances_to_building_blocks[m] for m in rxn.reactants
                    )

            if len(reactions_to_scores) == 0:
                raise RuntimeError("Cannot find a reaction backwards!")

            # we then pick the reaction with the lowest score -- this will be the one that results in the lowest overall
            # depth.
            min_highest_reaction = min(reactions_to_scores.keys(), key=reactions_to_scores.get)

            # we then add those reactants to our subsetted tree (adding them to the queue if they are not building
            # blocks, as then we'll also need to work out how to create it!).
            for reactant in min_highest_reaction.reactants:
                seen_mols.add(reactant)
                if distances_to_building_blocks[reactant] > 0:
                    # not a building block need to keep going!
                    mol_to_retrosynthesize_queue.append(reactant)

            seen_reactions.add(min_highest_reaction)

        distances = {m: distances_to_building_blocks[m] for m in seen_mols}
        yield self.subset_network_return_copy(seen_mols, seen_reactions), distances

    def get_molecule_distances_backwards_from_mol(
        self, mol_to_measure_from: MolecularNode, max_num_hops: int = 10
    ) -> dict[MolecularNode, numbers.Real]:
        """
        Goes backwards and sees what molecules are on the path to create the mol to measure from.

        Note that this function is helpful for visualizing etc, but that may not give most helpful score for working out
        how well a synthesis plan is near to completion (due to the "AND" nature of reaction distances).

        Dict only contains the molecules that are reachabnle within the max nuymber of hops.
        """
        node_distances = {mol_to_measure_from.canon_smi: 0}
        queue = collections.deque([(mol_to_measure_from, 0)])
        # ^ we will use this FIFO queue to keep a store of the molecules we have visited to work out which reactants
        # to process next.

        while queue:
            current_mol, current_dist = queue.popleft()

            # Once we've gone over the max number of hops we can break out as our job is done!
            if current_dist > max_num_hops:
                break

            # Try and find all the reactions where the current molecule is the product and process the reactants
            for rxn in self._mol_to_reactions[current_mol]:
                if rxn.product == current_mol:
                    new_dist = current_dist + 1
                    for reactant in rxn.reactants:
                        reactant_smi = reactant.canon_smi
                        if reactant_smi not in node_distances:
                            node_distances[reactant_smi] = new_dist
                            queue.append((reactant, new_dist))
                        else:
                            assert node_distances[reactant_smi] <= new_dist, (
                                "Distance should not be decreasing!"
                            )

        # Convert back to MolecularNode keys
        result = {self._canon_smi_to_molecule[smi]: dist for smi, dist in node_distances.items()}

        return result

    # @line_profiler.profile
    def get_molecule_distances_from_set(
        self, mol_set_to_measure_from: set[MolecularNode]
    ) -> dict[MolecularNode, numbers.Real]:
        """
        Returns a dictionary of distances from a set of molecules to all other molecules in terms of number of reaction
        hops (only look at hops in forward direction). Note the distance to a reaction is defined as the maximum
        distance over all its reactants. The distance to a molecule is the minimum distance over all reactions that
        produce it.

        (effectively variant of Dijkstra's algorithm.)
        """
        # implementation notes:
        # * switched to using canon_smi strs as the keys for molecular nodes as Python hashes these faster. (presumably
        #    due to some optimization).
        # * we don't fill in the node distances with infs at beginning any more as this caused the algorithm to be
        #    slower. (when finding next node in queue it had to search over unecessary nodes).

        # Set up storage for distances etc:
        working_node_distances: dict[str, float] = {
            m.canon_smi: 0.0 for m in mol_set_to_measure_from
        }
        edge_distances = {r: {m.canon_smi: float("inf") for m in r.reactants} for r in self.edges}
        # ^ a reaction has the distance which is the maximum to all of its reactants (i.e., an "and" operation).
        visited_nodes = {}

        with (
            tqdm.tqdm() as pbar
        ):  # <- we don't have a total as we might not need to visit all the nodes.
            while True:
                # Calculate the remaining nodes to visit. if this is none, we are done.
                if len(working_node_distances) == 0:
                    break

                # of the remaining nodes pick the closest one, if this is unreachable (i.e., inf) we are done.
                current = min(working_node_distances, key=working_node_distances.get)

                dist_to_current = working_node_distances.pop(current)
                visited_nodes[current] = dist_to_current
                # ^ mark the current node as visited and pop from working set so do not visit again.

                # for the visited node, update the distance to all reactions it partakes in.
                current_mol: MolecularNode = self._canon_smi_to_molecule[current]
                if current_mol in self._mol_to_reactions:
                    for r in self._mol_to_reactions[current_mol]:
                        if current == r.product.canon_smi:
                            continue
                        else:
                            assert current in edge_distances[r], (
                                "if not product it should be reactant!"
                            )

                        # - update distance to the visited node in reactant storage.
                        edge_distances[r][current] = (
                            dist_to_current  # <- now "visited" this distance is the minimum.
                        )

                        # - if all reactants now have distances, update the distance to the product.
                        # (note that this will always be plus one over the visited so okay to update this now).
                        product_reachable = not any(
                            d == float("inf") for d in edge_distances[r].values()
                        )
                        if product_reachable and (r.product.canon_smi not in visited_nodes):
                            working_node_distances[r.product.canon_smi] = min(
                                working_node_distances.get(r.product.canon_smi, float("inf")),
                                max(edge_distances[r].values()) + 1,
                            )
                            # ^ note product could have been visited by another reaction, so we take the minimum.
                pbar.update(1)

        # The ones we did not reach will have a distance of inf and switch all the keys back to MolecularNode objects.
        node_distances = {o: visited_nodes.get(o.canon_smi, float("inf")) for o in self.nodes}

        return node_distances

    @property
    def nodes(self) -> Iterable[MolecularNode]:
        """Iterator over the nodes (i.e., molecules) in the graph."""
        yield from self._canon_smi_to_molecule.values()

    @property
    def edges(self) -> Iterable[ReactionEdge]:
        """Iterator over the edges (i.e., reactions) in the graph."""
        yield from self._reactant_product_to_reaction.values()

    @property
    def num_edges(self) -> int:
        return len(self._reactant_product_to_reaction)

    @property
    def num_nodes(self) -> int:
        return len(self._canon_smi_to_molecule)

    def __eq__(self, other):
        molecules_same = self._canon_smi_to_molecule.keys() == other._canon_smi_to_molecule.keys()
        reactions_same = (
            self._reactant_product_to_reaction.keys() == other._reactant_product_to_reaction.keys()
        )
        return molecules_same and reactions_same

    def get_similarity_with_other(
        self, other: "Self", kind: Literal["molecules", "reactions", "both"] = "molecules"
    ) -> float:
        """Gets the fraction overlap of this reaction network with another. (i.e., the number of elements that are the
        same over the total number of unique elements -- same as Jaccard similarity.)

        if kind is molecules then only consider molecules, likewise for reactions. if both then considers both molecules
        and reactions.
        """
        if kind == "molecules":
            self_set = set(self._canon_smi_to_molecule.keys())
            other_set = set(other._canon_smi_to_molecule.keys())
        elif kind == "reactions":
            self_set = set(self._reactant_product_to_reaction.keys())
            other_set = set(other._reactant_product_to_reaction.keys())
        elif kind == "both":
            self_set = set(self._canon_smi_to_molecule.keys()).union(
                self._reactant_product_to_reaction.keys()
            )
            other_set = set(other._canon_smi_to_molecule.keys()).union(
                other._reactant_product_to_reaction.keys()
            )
        else:
            raise ValueError(f"Invalid kind: {kind}")

        intersection = self_set & other_set
        union = self_set | other_set
        if not union:
            return 1.0  # If both are empty, define overlap as 1.0
        return len(intersection) / len(union), len(intersection) / len(self_set)

    def get_subset_of_other(self, other: "Self") -> tuple[set[MolecularNode], set[ReactionEdge]]:
        """Gets components of this reaction network that are contained in the other."""
        # Find the set of MolecularNodes in self that are also in other
        common_molecules = set(self._canon_smi_to_molecule.keys()) & set(
            other._canon_smi_to_molecule.keys()
        )
        common_molecule_nodes = {self._canon_smi_to_molecule[smi] for smi in common_molecules}

        # Find the set of ReactionEdges in self that are also in other
        common_reactions = set(self._reactant_product_to_reaction.keys()) & set(
            other._reactant_product_to_reaction.keys()
        )
        common_reaction_edges = {
            self._reactant_product_to_reaction[key] for key in common_reactions
        }

        return common_molecule_nodes, common_reaction_edges

    def fraction_subset_of_other(
        self, other: "Self", kind: Literal["molecules", "reactions", "both"] = "molecules"
    ) -> float:
        """
        Returns the fraction of this reaction network that is contained in the other.
        """
        molecules, reactions = self.get_subset_of_other(other)

        if kind == "molecules":
            return len(molecules) / self.num_nodes
        elif kind == "reactions":
            return len(reactions) / self.num_edges
        elif kind == "both":
            return (len(molecules) + len(reactions)) / (self.num_nodes + self.num_edges)
        else:
            raise ValueError(f"Invalid kind: {kind}")
