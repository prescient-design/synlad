import copy

import torch
from rdkit import Chem

from synlad.data.components import synthesis_dataset as dataset
from synlad.data.components import synthesis_molecular_graphs as molecular_graphs
from synlad.models.components import synthesis_rxn_predictor as rxn_predictor
from synlad.tokenization import synthesis_serialization as serialization
from synlad.tokenization import synthesis_vocab as vocab


# torch_geometric.utils.smiles.from_rdmol whitelists num_radical_electrons in [0, 4]; predicted reaction
# products occasionally contain atoms outside that range (e.g. bare pentavalent centres like [P]/[As]),
# which raises ValueError during featurisation. Clamp to the whitelist to keep sampling running.
def _clamp_radical_electrons(mol: Chem.Mol) -> Chem.Mol:
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 4:
            atom.SetNumRadicalElectrons(4)
    return mol


class TokenActionHandler:
    """
    Handles picked actions, calling any external tools (e.g., reaction prediction), and working with the tokenizer and
    serializer to work out which next actions are possible.

    This class relies on serializer, tokenizer, RxnNetTaskBatch, and reaction prediction tools so is currently quite
    strongly coupled and complicated... Trying to encapsulate this complexity here for now, but maybe in future can
    think of a better solution!
    """

    def __init__(
        self,
        serializer: serialization.Serializer,
        rxn_predictor: rxn_predictor.ReactionPredictor,
    ):
        self.serializer = serializer
        self.rxn_predictor = rxn_predictor
        self.score_for_non_molecular_token = 0.0

    def _extract_molecules_from_reaction_sequences(
        self,
        current_sequences: dataset.RxnNetTaskBatch,
        actions: torch.TensorType,
        reaction_start_token: vocab.SPECIAL_TOKENS,
        reaction_delim_token: vocab.SPECIAL_TOKENS,
        func_to_create_rxn_predictor_input,
    ) -> list:
        """
        Extract molecules from reaction sequences (works for both forward and reverse reactions).

        Implementation note:
        - we have a vectorized way of picking out where the reaction relevant molecules lie in the action sequences.
        - this is all done using the torch tensors.
        - however, the actual extraction and turning into an object suitable for the tool is currently done in a for loop.
        - perhaps if we were doing lots of these we could consider parallelizing this step.
        - but this has to happen outside of torch tensors anyway currently, so this paralleization may rely on writing/
            using a more performant tokenizer class.
        - so will punt for now. and anyway, likely the time spent on this will be small compared to
            the time spend on reaction prediction/other parts of generation.
        - this method also has a few assertion (to double check format of sequences). these hopefully should never fail,
            and can likely be removed in future if want to skip these checks for speed.

        Args:
            current_sequences: Current batch sequences
            actions: Action tensor containing reaction tokens [B', S']
            reaction_start_token: Token that starts the reaction (e.g., FWD_RXN, REV_RXN)
            reaction_delim_token: Token that delimits the reaction (e.g., FWD_RXN_RUN, REV_RXN_RUN)

        Returns:
            List of RxnLMPredictionInput objects for reactions that need completion
        """
        reaction_delim_val = vocab.SPECIAL_TOKENS_LIBRARY.idx_frm_token(reaction_delim_token)
        reaction_start_val = vocab.SPECIAL_TOKENS_LIBRARY.idx_frm_token(reaction_start_token)

        num_reaction_delim_tokens = torch.sum(actions == reaction_delim_val, dim=1)
        # ^ note we do not need to check for the number in the current sequences, as this must be even as we always
        # assume that this class will have already completed them when they came up!

        # Check that we have either 0 or 1 delimiter tokens (indicating incomplete reaction)
        if not torch.isin(
            num_reaction_delim_tokens, torch.tensor([0, 1], device=num_reaction_delim_tokens.device)
        ).all():
            raise ValueError(
                f"Actions must contain either 0 or 1 {reaction_delim_token.value} tokens"
            )

        reactions_to_complete_mask = num_reaction_delim_tokens == 1
        if not reactions_to_complete_mask.any():
            return [], reactions_to_complete_mask

        # Get relevant sequences that need completion
        relevant_current_input_sequences = current_sequences.input_sequences[
            reactions_to_complete_mask
        ]
        relevant_action_sequences = actions[reactions_to_complete_mask]

        # Concatenate current sequences with actions
        all_relevant_actions = torch.cat(
            [relevant_current_input_sequences, relevant_action_sequences], dim=1
        )
        # ^ note this may end up putting some pad tokens between the old and the new, but this is fine as we'll
        # ignore them as we take out the relevant molecules

        # Reverse to find positions from the end (most recent tokens)
        all_relevant_actions_rev = torch.flip(all_relevant_actions, dims=[1])

        # Find the positions of start and delimiter tokens (from the end)
        start_rxn_position = torch.argmax(
            (all_relevant_actions_rev == reaction_start_val).int(), dim=1
        )
        # ^ note this will be the last position of any FWD_RXN tokens, as argmax picks the first occurence,
        # and we reversed the actions.
        end_rxn_position = torch.argmax(
            (all_relevant_actions_rev == reaction_delim_val).int(), dim=1
        )

        # Sanity check: delimiter should come after start token (in reversed sequence)
        assert torch.all(end_rxn_position < start_rxn_position), (
            f"Delimiter {reaction_delim_token.value} should come after start token {reaction_start_token.value}"
        )

        prediction_inputs = []

        for i, (start_idx, end_idx) in enumerate(
            zip(end_rxn_position, start_rxn_position, strict=False)
        ):
            start_idx = start_idx.item()
            end_idx = end_idx.item()

            # Extract molecule token indices between delimiter and start token
            molecule_seq_indcs = all_relevant_actions_rev[i, start_idx + 1 : end_idx]
            molecule_smiles = []

            for token_idx in molecule_seq_indcs:
                token_idx = token_idx.item()
                if token_idx == vocab.PAD_VALUE:
                    continue

                # Ensure this is a molecular token
                assert (
                    token_idx >= current_sequences.tkn_lib_collection.start_of_molecular_indices
                ), f"Expected molecular token, got {token_idx}"

                smiles = current_sequences.tkn_lib_collection.token_frm_idx(token_idx)
                molecule_smiles.append(smiles)

            # Create prediction input
            prediction_input = func_to_create_rxn_predictor_input(molecule_smiles)
            prediction_inputs.append(prediction_input)

        return prediction_inputs, reactions_to_complete_mask

    def __call__(
        self, current_sequences: dataset.CurrentSequences, actions: torch.TensorType
    ) -> dataset.RxnNetTaskBatch:
        """This takes the actions, completes them (if needed), and returns the new sequences.

        To break down all the things this method is responsible for:
        - runs any forward or backward reactions that are needed to complete the actions by calling appropiate tools.
        - adds any new graphs to the token library collection and graph collection.
        - computes the relevant masks.
        """
        if current_sequences.is_nested:
            raise NotImplementedError(
                "TokenActionHandler does not currently support nested tensors"
            )

        # 1. Run forward and reverse reactions to get predictions for any new molecules.
        # 1a Work out if any forward reactions are needed:
        fwd_rxn_prediction_inputs, fwd_rxn_masks = self._extract_molecules_from_reaction_sequences(
            current_sequences,
            actions,
            vocab.SPECIAL_TOKENS.FWD_RXN,
            vocab.SPECIAL_TOKENS.FWD_RXN_RUN,
            lambda x: rxn_predictor.RxnLMPredictionInput.from_reactant_smis(
                x, skip_canonicalization=True
            ),
        )
        fwd_indcs = torch.nonzero(fwd_rxn_masks, as_tuple=True)[0]

        # 1b Work out if any reverse reactions are needed:
        rev_rxn_prediction_inputs, rev_rxn_masks = self._extract_molecules_from_reaction_sequences(
            current_sequences,
            actions,
            vocab.SPECIAL_TOKENS.REV_RXN,
            vocab.SPECIAL_TOKENS.REV_RXN_RUN,
            lambda x: rxn_predictor.RxnLMPredictionInput.from_product_smi(
                x[0], skip_canonicalization=True
            ),
            # ^ note if multiple products are given, we take the first one.
        )
        rev_indcs = torch.nonzero(rev_rxn_masks, as_tuple=True)[0]
        indices_into_original_sequences = torch.cat([fwd_indcs, rev_indcs])

        # 1c Run the reactions:
        all_prediction_inputs = fwd_rxn_prediction_inputs + rev_rxn_prediction_inputs
        predictions = (
            self.rxn_predictor.predict(all_prediction_inputs)
            if len(all_prediction_inputs) > 0
            else []
        )
        predictions_exist_flag = len(predictions) > 0
        predictions = [
            next(iter(p)) for p in predictions
        ]  # take only the first prediction for each input.
        # ...atm we assume that there must be at least one prediction. in future we may want to allow for more than one.

        # 1d. tokenize the resulting results
        if predictions_exist_flag:
            max_num_of_new_molecules = max(len(p) for p in predictions)
            new_token_lib_collection = copy.copy(current_sequences.tkn_lib_collection)
            molecules_to_add_to_graphs = []
            new_molecules_indices = torch.full(
                (len(current_sequences.input_sequences), max_num_of_new_molecules + 1),
                dtype=torch.long,
                device=current_sequences.input_sequences.device,
                fill_value=vocab.PAD_VALUE,
            )
            # ^ note we add 1 to the max number of new molecules to allow us to put the end deliminator

            # convert predictions to indices, (note that any new molecules will be added to the end of the token library)
            new_molecules = []
            for i, prediction in enumerate(predictions):
                orig_i = indices_into_original_sequences[i].item()
                is_fwd_rxn = i < len(fwd_rxn_prediction_inputs)
                final_token = (
                    vocab.SPECIAL_TOKENS.FWD_RXN_RUN
                    if is_fwd_rxn
                    else vocab.SPECIAL_TOKENS.REV_RXN_RUN
                )
                for j, molecule in enumerate(list(prediction) + [final_token]):
                    if new_token_lib_collection.token_in_collection(molecule):
                        idx = new_token_lib_collection.idx_frm_token(molecule)
                    else:
                        idx = new_token_lib_collection.add_mol_token_to_end_and_get_idx(molecule)
                        molecules_to_add_to_graphs.append(molecule)
                        new_molecules.append(molecule)
                    new_molecules_indices[orig_i, j] = idx

            # 2. Update the graph collection
            new_graphs = [
                molecular_graphs.from_smiles(smi, mol_modififier=_clamp_radical_electrons).to(
                    current_sequences.device
                )
                for smi in new_molecules
            ]
            # Break up the current batch into individual graphs and add new ones
            if len(new_graphs) > 0:
                current_graph_list = molecular_graphs.to_data_list(current_sequences.graphs)
                all_graphs = current_graph_list + new_graphs
                updated_graph_batch = molecular_graphs.from_data_list(all_graphs)
            else:
                # No new graphs to add, keep the current batch
                updated_graph_batch = current_sequences.graphs

            num_new_molecules = len(new_molecules)
        else:
            num_new_molecules = 0
            # otherwise we don't need to update the token library collection or the graph collection
            new_token_lib_collection = current_sequences.tkn_lib_collection
            updated_graph_batch = current_sequences.graphs

        # 3. Update the sequences and from model masks
        current_sequence_lengths = torch.sum(
            current_sequences.input_sequences != vocab.PAD_VALUE, dim=1
        )
        action_lengths = torch.sum(actions != vocab.PAD_VALUE, dim=1)
        current_sequence_and_action_lengths = current_sequence_lengths + action_lengths

        if predictions_exist_flag:
            new_dim_1 = (
                current_sequences.input_sequences.shape[1]
                + actions.shape[1]
                + new_molecules_indices.shape[1]
            )
        else:
            new_dim_1 = current_sequences.input_sequences.shape[1] + actions.shape[1]

        new_input_sequences = torch.full(
            (current_sequences.input_sequences.shape[0], new_dim_1),
            vocab.PAD_VALUE,
            dtype=current_sequences.input_sequences.dtype,
            device=current_sequences.device,
        )
        new_input_sequences[:, : current_sequences.input_sequences.shape[1]] = (
            current_sequences.input_sequences
        )
        new_input_from_model_masks = torch.full_like(new_input_sequences, False, dtype=torch.bool)
        new_input_from_model_masks[:, : current_sequences.input_frm_model_masks.shape[1]] = (
            current_sequences.input_frm_model_masks
        )

        # -- Add the actions after the current sequences
        idx_for_actions = torch.arange(
            actions.shape[1], dtype=torch.long, device=current_sequences.input_sequences.device
        )[None, :].repeat(actions.shape[0], 1)
        idx_for_actions = idx_for_actions + current_sequence_lengths[:, None]
        new_input_sequences.scatter_(1, idx_for_actions, actions)
        new_input_from_model_masks.scatter_(1, idx_for_actions, actions != vocab.PAD_VALUE)
        # ^ all actions are from model, but padding are not actions from model.

        # -- Add the completion tokens (new molecules) after actions
        if predictions_exist_flag:
            idx_for_completions = torch.arange(
                new_molecules_indices.shape[1],
                dtype=torch.long,
                device=current_sequences.input_sequences.device,
            )[None, :].repeat(new_molecules_indices.shape[0], 1)
            idx_for_completions = idx_for_completions + current_sequence_and_action_lengths[:, None]
            new_input_sequences.scatter_(1, idx_for_completions, new_molecules_indices)
            new_input_from_model_masks.scatter_(1, idx_for_completions, False)
        # ^ note that this also is good for any padding which should also be False for these values.

        # -- remove any excess padding (as we were defensive when we created new_input_sequences)
        number_tokens_to_remove_from_end = torch.min(
            torch.sum(new_input_sequences == vocab.PAD_VALUE, dim=1)
        ).item()
        if number_tokens_to_remove_from_end > 0:
            new_dim_1 = new_input_sequences.shape[1] - number_tokens_to_remove_from_end
            new_input_sequences = new_input_sequences[:, :new_dim_1]
            new_input_from_model_masks = new_input_from_model_masks[:, :new_dim_1]
        old_seq_len_to_take_dim = min(new_dim_1, current_sequences.input_mol_scores.shape[1])
        # ^ note may not take all old sequences, if some are padding.

        # 4. Update the scores
        new_mol_scores = torch.full(
            (new_input_sequences.shape[0], new_dim_1),
            fill_value=self.score_for_non_molecular_token,
            dtype=torch.float,
            device=current_sequences.input_mol_scores.device,
        )
        new_mol_scores[:, :old_seq_len_to_take_dim] = current_sequences.input_mol_scores[
            :, :old_seq_len_to_take_dim
        ]

        # we will look at any of the new molecules that have been added and score them one by one. note that we currently
        # do this sequentially, but could consider parallelizing this in future.
        locs_to_mol_score = torch.nonzero(
            new_input_sequences >= new_token_lib_collection.start_of_molecular_indices
        )
        locs_to_mol_score_that_are_new = (
            locs_to_mol_score[:, 1] >= current_sequence_lengths[locs_to_mol_score[:, 0]]
        )
        locs_to_mol_score = locs_to_mol_score[locs_to_mol_score_that_are_new]
        for i, j in locs_to_mol_score:
            mol_scorer = current_sequences.get_mol_score_for_input_seq_idx(i)
            new_mol_scores[i, j] = mol_scorer(
                new_token_lib_collection.token_frm_idx(new_input_sequences[i, j])
            )

        # 5. Update the masks -- we'll do this by copying over the old masks and then running the serialization code
        # on the new added sequences. (This code is likely a bit inefficient at the moment, so can consider optimizing
        # if needed in future.)
        new_input_predictive_nxt_tkn_masks = torch.zeros(
            (
                *new_input_sequences.shape,
                current_sequences.input_predictive_nxt_tkn_masks.shape[-1] + num_new_molecules,
            ),
            dtype=torch.bool,
            device=current_sequences.input_predictive_nxt_tkn_masks.device,
        )
        # ^ note how the masks will get bigger if we have added new molecules to our library (but that they will)
        new_input_predictive_nxt_tkn_masks[
            :, :old_seq_len_to_take_dim, : current_sequences.input_predictive_nxt_tkn_masks.shape[2]
        ] = current_sequences.input_predictive_nxt_tkn_masks[:, :old_seq_len_to_take_dim]

        new_seqs_max_len = new_input_sequences.shape[1]
        for i, start_idx in enumerate(current_sequence_lengths):
            for j in range(start_idx, new_seqs_max_len):
                if new_input_sequences[i, j] == vocab.PAD_VALUE:
                    break  # making assumption that we will not (incorrectly) have any padding in the middle of a sequence.
                new_mask_ = serialization.Serializer.get_mask(
                    new_input_sequences[i, : j + 1].detach().cpu().numpy().tolist(),
                    new_token_lib_collection,
                )
                new_input_predictive_nxt_tkn_masks[i, j, :] = torch.tensor(
                    new_mask_,
                    dtype=torch.bool,
                    device=current_sequences.input_predictive_nxt_tkn_masks.device,
                )

        # 6. mol masks, and nonpad masks
        new_input_mol_masks = (
            new_input_sequences >= new_token_lib_collection.start_of_molecular_indices
        )
        new_input_nonpad_masks = new_input_sequences != vocab.PAD_VALUE

        # 7. Now can put this together and assemble the new dataset
        # note that this does not know about the scores, so is just a plain RxnNetTaskBatch rather than a CurrentSequences.
        new_sequences = dataset.RxnNetTaskBatch(
            tkn_lib_collection=new_token_lib_collection,
            graphs=updated_graph_batch,
            input_sequences=new_input_sequences,
            input_mol_scores=new_mol_scores,
            input_predictive_nxt_tkn_masks=new_input_predictive_nxt_tkn_masks,
            input_frm_model_masks=new_input_from_model_masks,
            input_mol_masks=new_input_mol_masks,
            input_nonpad_masks=new_input_nonpad_masks,
        )
        new_sequences.add_output_sequences()

        return new_sequences
