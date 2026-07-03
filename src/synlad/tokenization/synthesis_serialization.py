import copy
import functools

import numpy as np
from numpy import random as np_random
from numpy import typing as npt

from synlad.data.components import synthesis_reaction_network_graph as reaction_network_graph
from synlad.tokenization import synthesis_vocab as vocab


class RxnTypeWeighter:
    """Basic weighter that picks actions on predefined weight types."""

    def __init__(self, rxn_type_weights: dict[str, float]):
        self.rxn_type_weights = rxn_type_weights

    def __call__(self, action, *_):
        weight = self.rxn_type_weights.get(action[0], 1.0)
        return weight


class EarliestFirst:
    """Picks building blocks that make a difference (i.e., that are ready to react) as early as possible."""

    def __init__(self, bb_tkn_library: vocab.TokenLibrary):
        self.bb_tkn_library = bb_tkn_library

    def __call__(
        self, action, out_sequence, nodes_added_to_network, reactions_added_to_network, rxn_net
    ):
        # if a building block token, add if reactions it participates in do not require an intermediate node
        # that is yet to be added.
        if action[0] == vocab.SPECIAL_TOKENS.BB.value:
            bb_smi = action[1]
            bb_node = rxn_net._canon_smi_to_molecule[bb_smi]

            # Get all reactions involving this building block
            reactions = rxn_net._mol_to_reactions.get(bb_node, set())

            # Check each reaction where this BB is a reactant
            for rxn in reactions:
                if bb_node in rxn.reactants:
                    # Check if all other reactants are either building blocks or already added (then can do reaction)
                    other_reactants = rxn.reactants - {bb_node}

                    can_perform_reaction = all(
                        self.bb_tkn_library.token_in_library(r.canon_smi)
                        or r in nodes_added_to_network
                        for r in other_reactants
                    )

                    # If at least one reaction can be performed, return 1
                    if can_perform_reaction:
                        return 1

            # No valid reaction found, return 0
            return 0

        # Not a building block action, return 1 -- we don;t currently change these.
        return 1


class MultiplierWeighter:
    def __init__(self, weighters_to_multiply: list[callable]):
        self.weighters_to_multiply = weighters_to_multiply

    def __call__(self, *args, **kwargs):
        return functools.reduce(
            lambda x, y: x * y, (w(*args, **kwargs) for w in self.weighters_to_multiply)
        )


def dummy_mol_scorer(smi: str):
    return -1.0


RXN_TOKENS_VALS_TO_END_DELIM_VALS = {
    vocab.REV_RXN_RUN_VALUE: vocab.REV_RXN_RUN_VALUE,  # retrosynthesis deliminators end with REV_RXN_RUN
    vocab.FWD_RXN_RUN_VALUE: vocab.FWD_RXN_RUN_VALUE,  # forward reaction deliminators end with FWD_RXN_RUN
}


DEFAULT_SPECIAL_TKN_MOL_SCORE_VALUES = 0.0


class Serializer:
    """Serializes a reaction network and computes predictive masks."""

    def __init__(self, special_tkn_mol_score_values=DEFAULT_SPECIAL_TKN_MOL_SCORE_VALUES):
        """
        Keyword Arguments:
            special_tkn_mol_score_values -- value to use for inapplicable molecule scores (i.e., for the non-molecular
                tokens) (default: {None})
        """
        self.special_tkn_mol_score_values = special_tkn_mol_score_values

    def __call__(
        self,
        rxn_net: reaction_network_graph.ReactionNetwork,
        initial_seq: list[str],
        mol_scorer: callable,
        action_to_prob: callable,
        tkn_lib_collection: vocab.TokenLibraryCollection,
        rng: np_random.Generator | None = None,
        return_as_str: bool = False,
    ):
        """Serialize a reaction network graph into a sequence of tokens, i.e., integers. Also computes the masks for
        next tokens, a mask indicating whether output is predicted by main model or a synthesis oracle and a list of
        scores (floats for molecules, None for non applicable tokens.)

        Args:
            rxn_net (ReactionNetwork): The reaction network to serialize.
            initial_seq (list[str]): The initial sequence to start from. This should be a list of tokens in string form.
            mol_scorer (callable): A callable that takes a molecule SMILES string and returns a score.
            action_to_prob (callable): A function that takes in possible action, current seq, current nodes, and current
                edges and assigns a probability to that action.
            tkn_lib_collection (TokenLibraryCollection): The collection of token libaries defining the different token
                types and their respecitve integer ids.
            rng (np_random.Generator, optional): A random number generator. Will be created if not provided.
            return_as_str (bool, optional): If True, the output sequence will be in string form. Defaults to False.
        """
        if rng is None:
            rng = np_random.default_rng()

        # - set up some state variables to record how much of the network has currently been serialized.
        nodes_added_to_network = set()
        reactions_added_to_network = set()

        # - set up the output variables
        out_sequence = []
        out_masks = []
        out_predicted_by_model = []  # <- True if should be predicted by model. False if provided elsewhere (e.g, from
        # user via prompt or from a forward/backward reaction "oracle").
        out_scores = []  # <- molecule scores, e.g., property scores from a property predictor for optimization task.

        current_seq_to_process = copy.copy(initial_seq)

        # - define a function to check if serialization is complete (i.e., all reactions and nodes in the network have
        # been added).
        ALL_NODES_TO_ADD = set[reaction_network_graph.MolecularNode](rxn_net.nodes)
        ALL_EDGES_TO_ADD = set[reaction_network_graph.ReactionEdge](rxn_net.edges)

        def serialization_complete(nodes_added_to_network, reactions_added_to_network):
            return (
                nodes_added_to_network == ALL_NODES_TO_ADD
                and reactions_added_to_network == ALL_EDGES_TO_ADD
            )

        while len(current_seq_to_process) or not serialization_complete(
            nodes_added_to_network, reactions_added_to_network
        ):
            # If we have finished processing the current sequence, then we need to get a new one
            if len(current_seq_to_process) == 0:
                possible_next_actions = []

                # - compute the possible building block actions
                remaining_building_blocks = (
                    n.canon_smi for n in (ALL_NODES_TO_ADD - nodes_added_to_network)
                )
                remaining_building_blocks = (
                    s
                    for s in remaining_building_blocks
                    if tkn_lib_collection.bb_tkn_library.token_in_library(s)
                )
                # ^ filter for building block molecules only
                possible_next_actions.extend(
                    [[vocab.SPECIAL_TOKENS.BB.value, s] for s in remaining_building_blocks]
                )

                # - compute the possible forward reactions
                possible_reactions = (e for e in (ALL_EDGES_TO_ADD - reactions_added_to_network))
                possible_reactions = (
                    e
                    for e in possible_reactions
                    if set(e.reactants).issubset(nodes_added_to_network)
                )
                for rxn in possible_reactions:
                    reactants = [n.canon_smi for n in rxn.reactants]
                    rng.shuffle(reactants)
                    possible_next_actions.extend(
                        [
                            [
                                vocab.SPECIAL_TOKENS.FWD_RXN.value,
                                *reactants,
                                vocab.SPECIAL_TOKENS.FWD_RXN_RUN.value,
                                rxn.product.canon_smi,
                                vocab.SPECIAL_TOKENS.FWD_RXN_RUN.value,
                            ]
                        ]
                    )

                # - compute the possible retrosynthesis reactions
                possible_reactions = (e for e in (ALL_EDGES_TO_ADD - reactions_added_to_network))
                possible_reactions = (
                    e for e in possible_reactions if e.product in nodes_added_to_network
                )
                for rxn in possible_reactions:
                    reactants = [n.canon_smi for n in rxn.reactants]
                    rng.shuffle(reactants)
                    possible_next_actions.extend(
                        [
                            [
                                vocab.SPECIAL_TOKENS.REV_RXN.value,
                                rxn.product.canon_smi,
                                vocab.SPECIAL_TOKENS.REV_RXN_RUN.value,
                                *reactants,
                                vocab.SPECIAL_TOKENS.REV_RXN_RUN.value,
                            ]
                        ]
                    )

                # - pick one of these actions and add to the current sequence
                if len(possible_next_actions) == 0:
                    raise RuntimeError(
                        "No possible next actions -- ensure the network does not have dangling nodes!"
                    )
                else:
                    possible_next_action_probs = np.array(
                        [
                            action_to_prob(
                                a,
                                out_sequence,
                                nodes_added_to_network,
                                reactions_added_to_network,
                                rxn_net,
                            )
                            for a in possible_next_actions
                        ]
                    )
                    possible_next_action_probs = (
                        possible_next_action_probs / possible_next_action_probs.sum()
                    )
                    i_to_pick = rng.choice(len(possible_next_actions), p=possible_next_action_probs)
                    current_seq_to_process = possible_next_actions[i_to_pick]

            # Process the current sequence until empty.
            while len(current_seq_to_process):
                initial_token = current_seq_to_process[0]
                if initial_token not in vocab.SPECIAL_TOKENS:
                    raise ValueError(f"Unknown token: {initial_token}")
                token_e = vocab.SPECIAL_TOKENS(initial_token)  # convert back to enum

                if token_e is vocab.SPECIAL_TOKENS.RETRO:
                    # get the molecule to retrosynthesize
                    final_mol_smi = current_seq_to_process[1]
                    final_mol_node = rxn_net._canon_smi_to_molecule[final_mol_smi]
                    nodes_added_to_network.add(final_mol_node)

                    # get the stop token
                    stop_token = current_seq_to_process[2]
                    assert vocab.SPECIAL_TOKENS(stop_token) is vocab.SPECIAL_TOKENS.EOS, (
                        f"Expected EOS token, got {stop_token}"
                    )

                    seq_to_add = [
                        tkn_lib_collection.idx_frm_token(token_e.value),
                        tkn_lib_collection.idx_frm_token(final_mol_smi),
                        tkn_lib_collection.idx_frm_token(stop_token),
                    ]
                    out_sequence.extend(seq_to_add)
                    out_masks.extend(
                        [
                            self.get_mask(out_sequence[:-2], tkn_lib_collection),
                            self.get_mask(out_sequence[:-1], tkn_lib_collection),
                            self.get_mask(out_sequence, tkn_lib_collection),
                        ]
                    )
                    out_predicted_by_model.extend([False, False, False])
                    out_scores.extend(
                        [
                            self.special_tkn_mol_score_values,
                            mol_scorer(final_mol_smi),
                            self.special_tkn_mol_score_values,
                        ]
                    )
                    i = 3

                elif token_e is vocab.SPECIAL_TOKENS.BB:
                    # get the molecule to add to the network
                    new_mol_smi = current_seq_to_process[1]
                    new_mol_node = rxn_net._canon_smi_to_molecule[new_mol_smi]
                    nodes_added_to_network.add(new_mol_node)
                    out_sequence.extend(
                        [
                            tkn_lib_collection.idx_frm_token(token_e.value),
                            tkn_lib_collection.idx_frm_token(new_mol_smi),
                        ]
                    )

                    out_masks.extend(
                        [
                            self.get_mask(out_sequence[:-1], tkn_lib_collection),
                            self.get_mask(out_sequence, tkn_lib_collection),
                        ]
                    )
                    out_predicted_by_model.extend([True, True])
                    out_scores.extend([self.special_tkn_mol_score_values, mol_scorer(new_mol_smi)])
                    i = 2

                elif token_e is vocab.SPECIAL_TOKENS.FWD_RXN:
                    # Add the outputs corresponding to the first token (i.e. forward reaction)
                    out_scores.append(
                        self.special_tkn_mol_score_values
                    )  # <- not a molecular token so cannot be scored
                    out_predicted_by_model.append(True)  # <- should be predicted by model
                    out_sequence.append(tkn_lib_collection.idx_frm_token(token_e.value))
                    out_masks.append(self.get_mask(out_sequence, tkn_lib_collection))

                    # Step through the tokens descibing the rest of the reaction, registering the respective outputs and
                    # also the different parts of reaction.
                    reactants = set()
                    products = set()
                    num_forward_reaction_delims_seen = 0  # counts of FWD_RXN_RUN
                    for j, token in enumerate(current_seq_to_process[1:], start=1):
                        # can add the token and its mask regardless of type
                        out_sequence.append(tkn_lib_collection.idx_frm_token(token))
                        out_masks.append(self.get_mask(out_sequence, tkn_lib_collection))

                        # - if a FWD_RXN_RUN token then we need to work out if first or second
                        if token == vocab.SPECIAL_TOKENS.FWD_RXN_RUN:
                            out_scores.append(
                                self.special_tkn_mol_score_values
                            )  # <- for reaction forward token

                            num_forward_reaction_delims_seen += 1
                            if num_forward_reaction_delims_seen == 1:
                                out_predicted_by_model.append(
                                    True
                                )  # <- first reaction token given by model
                            elif num_forward_reaction_delims_seen == 2:
                                out_predicted_by_model.append(
                                    False
                                )  # <- second reaction token given by oracle.
                                length_to_use = j + 1
                                break  # <- can break out now too as reaction must be over
                            else:
                                raise RuntimeError("More than two FWD_RXN_RUN seen!")

                        # - else must be a molecular token.
                        else:
                            if num_forward_reaction_delims_seen == 0:
                                reactants.add(token)
                                out_scores.append(mol_scorer(token))
                                out_predicted_by_model.append(True)
                            elif num_forward_reaction_delims_seen == 1:
                                products.add(token)
                                out_scores.append(mol_scorer(token))
                                out_predicted_by_model.append(
                                    False
                                )  # <- these will be filled in by a separate "oracle"
                    else:
                        raise RuntimeError("Fwd reaction not closed!")

                    # - get the parts according to the reaction network and record these
                    reactant_nodes = {rxn_net._canon_smi_to_molecule[r] for r in reactants}
                    product_nodes = {rxn_net._canon_smi_to_molecule[p] for p in products}
                    assert len(product_nodes) == 1, "Currently assuming one product per reaction."
                    reaction_edge = rxn_net._reactant_product_to_reaction[
                        (frozenset(reactant_nodes), next(iter(product_nodes)))
                    ]
                    reactions_added_to_network.add(reaction_edge)
                    assert reactant_nodes.issubset(nodes_added_to_network), (
                        "adding a reaction before nodes exist!"
                    )
                    nodes_added_to_network.update(product_nodes)

                    # finally record how much of the current processing seq was used
                    i = length_to_use

                elif token_e is vocab.SPECIAL_TOKENS.REV_RXN:
                    # Add the outputs corresponding to the first token (i.e. backward reaction)
                    out_scores.append(
                        self.special_tkn_mol_score_values
                    )  # <- not a molecular token so cannot be scored
                    out_predicted_by_model.append(True)  # <- should be predicted by model
                    out_sequence.append(tkn_lib_collection.idx_frm_token(token_e.value))
                    out_masks.append(self.get_mask(out_sequence, tkn_lib_collection))

                    # step through the tokens describing the rest of the reaction, registering the products and reactants
                    reactants = set()
                    products = set()
                    num_forward_reaction_delims_seen = 0  # counts of REV_RXN_RUN
                    for j, token in enumerate(current_seq_to_process[1:], start=1):
                        # can add the token and its mask regardless of type
                        out_sequence.append(tkn_lib_collection.idx_frm_token(token))
                        out_masks.append(self.get_mask(out_sequence, tkn_lib_collection))

                        # - if a REV_RXN_RUN token then we need to work out if first or second
                        if token == vocab.SPECIAL_TOKENS.REV_RXN_RUN:
                            out_scores.append(
                                self.special_tkn_mol_score_values
                            )  # <- for reaction forward token

                            num_forward_reaction_delims_seen += 1
                            if num_forward_reaction_delims_seen == 1:
                                out_predicted_by_model.append(
                                    True
                                )  # <- first reaction token given by model
                            elif num_forward_reaction_delims_seen == 2:
                                out_predicted_by_model.append(
                                    False
                                )  # <- second reaction token given by oracle.
                                length_to_use = j + 1
                                break  # <- can break out now too as reaction must be over
                            else:
                                raise RuntimeError("More than two REV_RXN_RUN seen!")

                        # - else must be a molecular token.
                        else:
                            if num_forward_reaction_delims_seen == 0:
                                products.add(token)
                                out_scores.append(mol_scorer(token))
                                out_predicted_by_model.append(True)
                                assert len(products) < 2, (
                                    "only single product retrosytnthesis currently supported"
                                )
                            elif num_forward_reaction_delims_seen == 1:
                                reactants.add(token)
                                out_scores.append(mol_scorer(token))
                                out_predicted_by_model.append(
                                    False
                                )  # <- these will be filled in by a separate "oracle"
                    else:
                        raise RuntimeError("Backward reaction not closed!")

                    # - get the parts according to the reaction network and record these
                    reactant_nodes = {rxn_net._canon_smi_to_molecule[r] for r in reactants}
                    product_nodes = {rxn_net._canon_smi_to_molecule[p] for p in products}
                    reaction_edge = rxn_net._reactant_product_to_reaction[
                        (frozenset(reactant_nodes), next(iter(product_nodes)))
                    ]
                    reactions_added_to_network.add(reaction_edge)
                    assert product_nodes.issubset(nodes_added_to_network), (
                        "adding a retrosynthesis before product node exists!"
                    )
                    nodes_added_to_network.update(reactant_nodes)

                    # finally record how much of the current processing seq was used
                    i = length_to_use

                else:
                    raise NotImplementedError(
                        f"Serialization with {initial_token} not implemented yet!"
                    )

                current_seq_to_process = current_seq_to_process[i:]

        # add the final EOS token
        out_sequence.append(
            tkn_lib_collection.special_tkn_library.idx_frm_token(vocab.SPECIAL_TOKENS.EOS)
        )
        out_masks.append(self.get_mask(out_sequence, tkn_lib_collection))
        out_predicted_by_model.append(True)
        out_scores.append(self.special_tkn_mol_score_values)

        # if requested as string convert back (this is slightly wasteful, could optimize if end up refactoring)
        if return_as_str:
            out_sequence = [tkn_lib_collection.token_frm_idx(i) for i in out_sequence]

        return out_sequence, out_masks, out_predicted_by_model, out_scores

    @classmethod
    def get_if_from_model(
        cls, sequence: list[int | str], tkn_lib_collection: vocab.TokenLibraryCollection
    ) -> np.array:
        """Return a boolean mask indicating if the token should be predicted by the model or not."""
        out = []
        current_val = True  # <- records if next token should be generated by model (assuming no information otherwise)
        end_tkn = None
        for tkn in sequence:
            if isinstance(tkn, str):
                tkn = tkn_lib_collection.idx_frm_token(tkn)

            # pad tokens are never predicted by model.
            if tkn == vocab.PAD_VALUE:
                out.append(False)
                continue

            # if currently coming from user/tool and the corresponding end user/tool token is seen then we are back to
            # model-generated tokens.
            if not current_val and tkn == end_tkn:
                out.append(current_val)  # but only change after!
                current_val = True
            # if a prefix token is seen then we are user generated tokens (including the prefix token) until EOS is seen.
            elif tkn in vocab.PREFIX_TOKEN_VALUES:
                current_val = False
                end_tkn = vocab.EOS_VALUE
                out.append(False)
            # if a tool token is seen (i.e., rxn) then we are tool (i.e., not model-generated) tokens until the relevant
            # end tool token is seen.
            elif tkn in RXN_TOKENS_VALS_TO_END_DELIM_VALS:
                assert current_val, "Expected model-generated token for tool innovation token!"
                out.append(True)  # <- note True as tool innovation should be from model
                current_val = False
                end_tkn = RXN_TOKENS_VALS_TO_END_DELIM_VALS[tkn]
            else:
                out.append(current_val)
        return np.array(out, dtype=np.bool_)

    @classmethod
    def get_mask(
        cls, seq_so_far: list[str | int], tkn_lib_collection: vocab.TokenLibraryCollection
    ) -> np.array:
        """Gets the mask for the next token in the sequence.


        Args:
            seq_so_far (list): The sequence so far -- can be either in integer or string form (but not mixed), we will
                infer based on the first token.
            tkn_lib_collection (TokenLibraryCollection): The collection of token libaries defining the different token
                types and their respecitve integer ids.
        """
        # if given tokens as strings then convert to integers
        if len(seq_so_far) and isinstance(seq_so_far[0], str):
            seq_so_far = [tkn_lib_collection.idx_frm_token(t) for t in seq_so_far]

        # create mask of all zeros which we will place the positive values into.
        mask = np.zeros(tkn_lib_collection.end_idx + 1, dtype=np.bool_)

        # 1. empty seq can start with reaction network tokens (prompt-free generation)
        if len(seq_so_far) == 0:
            rxnnet_tokens_indcs = tkn_lib_collection.special_tkn_library.token_indcs(
                vocab.RXNNET_TOKENS
            )
            mask[rxnnet_tokens_indcs] = True
            return mask

        # otherwise find last main action token
        for i, token_idx in zip(
            range(len(seq_so_far) - 1, -1, -1), reversed(seq_so_far), strict=False
        ):
            token_val = tkn_lib_collection.token_frm_idx(token_idx)
            if token_val in vocab.MAIN_ACTION_TOKEN:
                last_main_action_token = token_val
                last_main_seq_idx = i
                break
        else:
            raise RuntimeError("No main action token found in sequence!")

        last_main_action_token_e = vocab.SPECIAL_TOKENS(last_main_action_token)
        current_seq_len = len(seq_so_far)
        tokens_after = current_seq_len - last_main_seq_idx - 1

        # 2. Retro prompt
        if last_main_action_token_e is vocab.SPECIAL_TOKENS.RETRO:
            # a. stop token
            if tokens_after == 1:
                mask[
                    tkn_lib_collection.special_tkn_library.idx_frm_token(vocab.SPECIAL_TOKENS.EOS)
                ] = True

            # b. molecule token (defining molecule to synthesize)
            elif tokens_after == 0:
                mask[
                    tkn_lib_collection.mol_tkn_library.start_idx : tkn_lib_collection.mol_tkn_library.end_idx
                    + 1
                ] = True
                # ^ currently wouldn't bother doing retro on a building block.
            else:
                raise RuntimeError("Invalid sequence after RETRO token!")

        # EOS
        elif last_main_action_token_e is vocab.SPECIAL_TOKENS.EOS:
            rxnnet_tokens_indcs = tkn_lib_collection.special_tkn_library.token_indcs(
                vocab.RXNNET_TOKENS
            )
            mask[rxnnet_tokens_indcs] = True

        # 3. Building block prompt
        elif last_main_action_token_e is vocab.SPECIAL_TOKENS.BB:
            # a. new rxn net action
            if tokens_after == 1:
                rxnnet_tokens_indcs = tkn_lib_collection.special_tkn_library.token_indcs(
                    vocab.RXNNET_TOKENS
                )
                mask[rxnnet_tokens_indcs] = True
                mask[
                    tkn_lib_collection.special_tkn_library.idx_frm_token(vocab.SPECIAL_TOKENS.EOS)
                ] = True

            # b. new molecule token (only from building block set)
            if tokens_after == 0:
                mask[
                    tkn_lib_collection.bb_tkn_library.start_idx : tkn_lib_collection.bb_tkn_library.end_idx
                    + 1
                ] = True

        # 4. Forward reaction token
        elif last_main_action_token_e is vocab.SPECIAL_TOKENS.FWD_RXN:
            # a. first one after can be any created molecule (only).
            if tokens_after == 0:
                mask[cls._get_idncs_of_mols_created_so_far(seq_so_far, tkn_lib_collection)] = True

            # otherwise depends on whether we have finished predicting reactants and product
            else:
                last_tkn_m1 = tkn_lib_collection.token_frm_idx(seq_so_far[-1])
                num_rxn_forward_tokens = sum(
                    1
                    for tkn in seq_so_far[last_main_seq_idx:]
                    if tkn
                    == tkn_lib_collection.idx_frm_token(vocab.SPECIAL_TOKENS.FWD_RXN_RUN.value)
                )

                # b. if have predicted the second FWD_RXN_RUN token then we are ready to add more rxn net actions.
                if (
                    last_tkn_m1 == vocab.SPECIAL_TOKENS.FWD_RXN_RUN.value
                    and num_rxn_forward_tokens == 2
                ):
                    rxnnet_tokens_indcs = tkn_lib_collection.special_tkn_library.token_indcs(
                        vocab.RXNNET_TOKENS
                    )
                    mask[rxnnet_tokens_indcs] = True
                    mask[
                        tkn_lib_collection.special_tkn_library.idx_frm_token(
                            vocab.SPECIAL_TOKENS.EOS
                        )
                    ] = True

                # c. if we have only just added the first FWD_RXN_RUN token then we can only add products (they can be
                # any).
                elif (
                    last_tkn_m1 == vocab.SPECIAL_TOKENS.FWD_RXN_RUN.value
                    and num_rxn_forward_tokens == 1
                ):
                    mask[tkn_lib_collection.start_of_molecular_indices :] = True
                    mask[
                        tkn_lib_collection.special_tkn_library.idx_frm_token(
                            vocab.SPECIAL_TOKENS.FWD_RXN_RUN
                        )
                    ] = True
                    # ^ can also add second deliminator (eg if reaction did not produce anything)

                # d. if no num_rxn_forward_tokens we can add either an existing molecule or a FWD_RXN_RUNd token
                elif num_rxn_forward_tokens == 0:
                    mask[cls._get_idncs_of_mols_created_so_far(seq_so_far, tkn_lib_collection)] = (
                        True
                    )
                    mask[
                        tkn_lib_collection.special_tkn_library.idx_frm_token(
                            vocab.SPECIAL_TOKENS.FWD_RXN_RUN
                        )
                    ] = True

                # e. otherwise we can add either a molecule (any -- as must be product) or a FWD_RXN_RUN token
                else:
                    mask[tkn_lib_collection.start_of_molecular_indices :] = True
                    mask[
                        tkn_lib_collection.special_tkn_library.idx_frm_token(
                            vocab.SPECIAL_TOKENS.FWD_RXN_RUN
                        )
                    ] = True

        # 5. Reverse reaction token
        elif last_main_action_token_e is vocab.SPECIAL_TOKENS.REV_RXN:
            # a. first one after can be any created molecule (only).
            if tokens_after == 0:
                mask[cls._get_idncs_of_mols_created_so_far(seq_so_far, tkn_lib_collection)] = True

            # b. if added a molecule then we must do the REV_RXN_RUN token afterwards (can only retrosynthesize
            # a single product).)
            elif tokens_after == 1:
                mask[
                    tkn_lib_collection.special_tkn_library.idx_frm_token(
                        vocab.SPECIAL_TOKENS.REV_RXN_RUN
                    )
                ] = True

            else:
                last_tkn_m1 = tkn_lib_collection.token_frm_idx(seq_so_far[-1])
                num_rxn_reverse_tokens = sum(
                    1
                    for tkn in seq_so_far[last_main_seq_idx:]
                    if tkn
                    == tkn_lib_collection.idx_frm_token(vocab.SPECIAL_TOKENS.REV_RXN_RUN.value)
                )

                # c. if have predicted the second REV_RXN_RUN token then we are ready to add more rxn net actions.
                if (
                    last_tkn_m1 == vocab.SPECIAL_TOKENS.REV_RXN_RUN.value
                    and num_rxn_reverse_tokens == 2
                ):
                    rxnnet_tokens_indcs = tkn_lib_collection.special_tkn_library.token_indcs(
                        vocab.RXNNET_TOKENS
                    )
                    mask[rxnnet_tokens_indcs] = True
                    mask[
                        tkn_lib_collection.special_tkn_library.idx_frm_token(
                            vocab.SPECIAL_TOKENS.EOS
                        )
                    ] = True

                # d. if we have only just added the first reaction forward token then we can only add products -- and
                # they dont have to be already seen.
                elif (
                    last_tkn_m1 == vocab.SPECIAL_TOKENS.REV_RXN_RUN.value
                    and num_rxn_reverse_tokens == 1
                ):
                    mask[tkn_lib_collection.start_of_molecular_indices :] = True
                    mask[
                        tkn_lib_collection.special_tkn_library.idx_frm_token(
                            vocab.SPECIAL_TOKENS.REV_RXN_RUN
                        )
                    ] = True
                    # ^ can also add second deliminator, e.g., if retrosynthesis did not produce anything.

                # e. otherwise we can add either a molecule (any) or a REV_RXN_RUN token
                else:
                    mask[tkn_lib_collection.start_of_molecular_indices :] = True
                    mask[
                        tkn_lib_collection.special_tkn_library.idx_frm_token(
                            vocab.SPECIAL_TOKENS.REV_RXN_RUN
                        )
                    ] = True

        else:
            raise NotImplementedError(f"Masking with {last_main_action_token} not implemented yet!")

        return mask

    @classmethod
    def _get_idncs_of_mols_created_so_far(
        cls, seq_so_far: list[int], tkn_lib_collection: vocab.TokenLibraryCollection
    ) -> npt.NDArray[np.int_]:
        out = [
            i
            for i in seq_so_far
            if tkn_lib_collection.bb_tkn_library.start_idx
            <= i
            <= tkn_lib_collection.mol_tkn_library.end_idx
        ]
        return np.array(out)


class Deserializer:
    def __init__(self, add_prompt_molecules_to_graph: bool = False):
        if add_prompt_molecules_to_graph:
            raise NotImplementedError("Adding prompt molecules to graph is not implemented yet!")

    @staticmethod
    def _convert_int_tokens_to_str_if_needed(
        seq_to_deserialize: list[int | str], tkn_lib_collection: vocab.TokenLibraryCollection | None
    ) -> list[str]:
        if len(seq_to_deserialize) and isinstance(seq_to_deserialize[0], int):
            assert tkn_lib_collection is not None, (
                "tkn_lib_collection must be provided if given tokens as integers"
            )
            return [tkn_lib_collection.token_frm_idx(i) for i in seq_to_deserialize]
        return seq_to_deserialize

    def __call__(
        self,
        seq_to_deserialize: list[int | str],
        tkn_lib_collection: vocab.TokenLibraryCollection | None = None,
    ):
        """Deserialize a sequence of tokens into a reaction network graph.

        Args:
            seq (list[int | str]): The sequence to deserialize. This should be a list of tokens in string or integer
                form. Will infer from first token whether in str or int form. (cannot mix)
            tkn_lib_collection (TokenLibraryCollection): The collection of token libaries defining the different token
                types and their respecitve integer ids.
        """
        seq_to_deserialize = self._convert_int_tokens_to_str_if_needed(
            seq_to_deserialize, tkn_lib_collection
        )

        # Deserialize!
        rxn_net = reaction_network_graph.ReactionNetwork()
        prompt = []
        mols_explored = {}  # <- dict to hold the molecules we have seen so far -- using as set-like but also ordered in
        # Python 3.7+. values are unimportant.

        reaction_order = 0

        while len(seq_to_deserialize):
            initial_token = seq_to_deserialize[0]
            if initial_token not in vocab.SPECIAL_TOKENS:
                raise ValueError(f"Unknown token: {initial_token}")
            token_e = vocab.SPECIAL_TOKENS(initial_token)

            if token_e in vocab.PREFIX_TOKENS:
                for i, val in enumerate(seq_to_deserialize):  # noqa: B007
                    prompt.append(val)
                    if val == vocab.SPECIAL_TOKENS.EOS.value:
                        break
                num_indcs_processed = i + 1

            elif token_e is vocab.SPECIAL_TOKENS.EOS:
                # this doesn't change the graph so nothing happens...
                num_indcs_processed = 1

            elif token_e is vocab.SPECIAL_TOKENS.BB:
                mols_explored[seq_to_deserialize[1]] = None
                rxn_net.add_smi_sets(
                    [seq_to_deserialize[1]], canonicalize=False
                )  # add node to graph
                num_indcs_processed = 2

            elif token_e is vocab.SPECIAL_TOKENS.FWD_RXN:
                num_fwd_delims_seen = 0
                reactants = []
                products = []
                for i, val in enumerate(seq_to_deserialize[1:], start=1):  # noqa: B007
                    if val == vocab.SPECIAL_TOKENS.FWD_RXN_RUN.value:
                        num_fwd_delims_seen += 1
                        if num_fwd_delims_seen == 2:
                            break
                    else:
                        if num_fwd_delims_seen == 0:
                            reactants.append(val)
                        elif num_fwd_delims_seen == 1:
                            products.append(val)
                            mols_explored[val] = None
                else:
                    raise RuntimeError("FWD_RXN not closed!")

                assert len(products) == 1, "Currently assuming one product per reaction."
                rxn_net.add_reaction_smi_sets(
                    reactants,
                    products[0],
                    lambda d_: d_.update({"direction": "fwd", "reaction_order": reaction_order}),
                    canonicalize=False,
                )
                num_indcs_processed = i + 1

            elif token_e is vocab.SPECIAL_TOKENS.REV_RXN:
                num_rev_delims_seen = 0
                reactants = []
                products = []
                for i, val in enumerate(seq_to_deserialize[1:], start=1):  # noqa: B007
                    if val == vocab.SPECIAL_TOKENS.REV_RXN_RUN.value:
                        num_rev_delims_seen += 1
                        if num_rev_delims_seen == 2:
                            break
                    else:
                        if num_rev_delims_seen == 0:
                            products.append(val)
                        elif num_rev_delims_seen == 1:
                            reactants.append(val)
                            if val not in mols_explored:
                                mols_explored[val] = None
                else:
                    raise RuntimeError("REV_RXN not closed!")

                rxn_net.add_reaction_smi_sets(
                    reactants,
                    products[0],
                    lambda d_: d_.update({"direction": "rev", "reaction_order": reaction_order}),
                    canonicalize=False,
                )
                num_indcs_processed = i + 1

            else:
                raise NotImplementedError(
                    f"Deserialization with {initial_token} not implemented yet!"
                )

            seq_to_deserialize = seq_to_deserialize[num_indcs_processed:]

        # convert mols_explored to a list
        mols_explored = list(mols_explored.keys())

        return rxn_net, prompt, mols_explored

    @classmethod
    def break_sequence_up_into_node_adding_subsequences(cls, seq_to_break: list[int | str]):
        """Breaks up a linear sequence into a list of lists, where each list is a subsequence describing the molecule
        added to a graph.

        For instance:
            [<RETRO>, <MOL1>, <EOS>, <BB>, <MOL2>, <FWD_RXN>, <MOL3>, <FWD_RXN_RUN>, <MOL4> <FWD_RXN_RUN> <EOS>]
        would be broken up into:
            [[<RETRO>, <MOL1>], [<EOS>], [<BB>, <MOL2>], [<FWD_RXN>, <MOL3>, <FWD_RXN_RUN>, <MOL4>, <FWD_RXN_RUN>], [<EOS>]]

        If last token set it incomplete then will get treated as seperate list even if not yet complete.
        """
        if isinstance(seq_to_break[0], int):
            assert all(isinstance(t, int) for t in seq_to_break), (
                "cannot mix string and integer tokens"
            )
            tkns_to_break_on = {
                vocab.SPECIAL_TOKENS_LIBRARY.idx_frm_token(t) for t in vocab.MAIN_ACTION_TOKEN
            }
        else:
            assert all(isinstance(t, str) for t in seq_to_break), (
                "cannot mix string and integer tokens"
            )
            tkns_to_break_on = vocab.MAIN_ACTION_TOKEN

        out = []
        current_subsequence = []
        for tkn in seq_to_break:
            if tkn in tkns_to_break_on:
                if current_subsequence:
                    out.append(current_subsequence)
                current_subsequence = []
            current_subsequence.append(tkn)
        out.append(current_subsequence)
        return out

    @staticmethod
    def _parse_reaction_block(
        block: list[str], is_forward: bool
    ) -> tuple[list[str], list[str], list[str]]:
        """Parse a reaction block into reactants, products, and new molecules."""
        delimiter = (
            vocab.SPECIAL_TOKENS.FWD_RXN_RUN.value
            if is_forward
            else vocab.SPECIAL_TOKENS.REV_RXN_RUN.value
        )
        first_group, second_group = [], []
        delim_count = 0

        for token in block[1:]:
            if token == delimiter:
                delim_count += 1
                if delim_count >= 2:
                    break
            elif delim_count == 0:
                first_group.append(token)
            elif delim_count == 1:
                second_group.append(token)

        reactants = first_group if is_forward else second_group
        products = second_group if is_forward else first_group

        return reactants, products, second_group

    @classmethod
    def get_molecule_reaction_first_indices(
        cls, seq_to_process: list[int | str], tkn_lib_collection: vocab.TokenLibraryCollection
    ) -> tuple[dict[str, int], dict[reaction_network_graph.ReactionType, int]]:
        """Parses a sequence of tokens and returns the indices (index from 0) of where the first occurence of each molecule occurs,
        as well as the indices of where each reaction starts. Reverse and forward reactions are treated the same, and
        reverse reactions are converted into a forward representation for the output.

        For instance:
            [<RETRO>, <MOL1>, <EOS>, <BB>, <MOL2>, <FWD_RXN>, <MOL2>, <FWD_RXN_RUN>, <MOL1> <FWD_RXN_RUN> <EOS>]
        would be broken up into:
        mol_dict = {<MOL1>: 1, <MOL2>: 4}
        rxn_dict = {(frozenset({<MOL2>}), <MOL1>): 5}

        Args:
            seq_to_process (list[int | str]): The sequence to process. This should be a list of tokens in string or integer
                form. Will infer from first token whether in str or int form. (cannot mix)
            tkn_lib_collection (TokenLibraryCollection): The collection of token libaries defining the different token
                types and their respecitve integer ids.
        Returns:
            Tuple[Dict[str, int], Dict[reaction_network_graph.ReactionType, int]]: The indices of where the first occurence of
                each molecule occurs, as well as the indices of where each reaction starts. Molecules are represented
                by their canonical SMILES>.
        """
        seq_to_process = cls._convert_int_tokens_to_str_if_needed(
            seq_to_process, tkn_lib_collection
        )
        blocks = cls.break_sequence_up_into_node_adding_subsequences(seq_to_process)

        mol_first_indices: dict[str, int] = {}
        rxn_first_indices: dict[reaction_network_graph.ReactionType, int] = {}
        cumulative_idx = 0

        for block in blocks:
            first_token = block[0]
            if first_token not in vocab.SPECIAL_TOKENS:
                raise ValueError(f"Unknown token: {first_token}")
            token_e = vocab.SPECIAL_TOKENS(first_token)

            if token_e is vocab.SPECIAL_TOKENS.RETRO:
                mol_smi = block[1]
                if mol_smi not in mol_first_indices:
                    mol_first_indices[mol_smi] = cumulative_idx + 1

            elif token_e is vocab.SPECIAL_TOKENS.BB:
                mol_smi = block[1]
                if mol_smi not in mol_first_indices:
                    mol_first_indices[mol_smi] = cumulative_idx + 1

            elif token_e in (vocab.SPECIAL_TOKENS.FWD_RXN, vocab.SPECIAL_TOKENS.REV_RXN):
                is_forward = token_e is vocab.SPECIAL_TOKENS.FWD_RXN
                reactants, products, new_mols = cls._parse_reaction_block(block, is_forward)

                for mol_smi in new_mols:
                    if mol_smi not in mol_first_indices:
                        mol_idx_in_block = block.index(mol_smi, 1)
                        mol_first_indices[mol_smi] = cumulative_idx + mol_idx_in_block

                assert len(products) == 1, (
                    "currently only single product retrosynthesis is supported"
                )
                rxn_key = (frozenset(reactants), products[0])
                if rxn_key not in rxn_first_indices:
                    rxn_first_indices[rxn_key] = cumulative_idx

            elif token_e is vocab.SPECIAL_TOKENS.EOS:
                pass

            else:
                raise NotImplementedError(f"Token {token_e} not implemented yet!")

            cumulative_idx += len(block)

        return mol_first_indices, rxn_first_indices
