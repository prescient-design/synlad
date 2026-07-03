import enum
from collections.abc import Callable

import torch
import torch_geometric.data as pyg_data

from synlad.data.components import synthesis_dataset as dataset
from synlad.data.components import synthesis_molecular_graphs as molecular_graphs
from synlad.models import synthesis_decoder
from synlad.models.components import synthesis_inference_strategies as inference_strategies
from synlad.models.components import synthesis_token_action_handler as token_action_handler
from synlad.tokenization import synthesis_serialization as serialization
from synlad.tokenization import synthesis_vocab as vocab


class Pipeline:
    """
    Pipeline for generating completions from a model.
    """

    def __init__(
        self,
        generator: inference_strategies.InferenceStrategy,
        tkn_lib_collection: vocab.TokenLibraryCollection,
        device: torch.device,
        bb_graph_list: list[pyg_data.Data] | None = None,  # if None, we will recreate!
        score_normalizer: Callable[[tuple[int, ...], float], float] | None = None,
        deserializer: serialization.Deserializer | None = None,
        top_k: int = 5,
    ):
        self.generator = generator
        self.tkn_lib_collection = tkn_lib_collection
        if bb_graph_list is None:
            bb_graph_list = [
                molecular_graphs.from_smiles(smi)
                for smi in self.tkn_lib_collection.bb_tkn_library.tokens
            ]
        self.bb_graph_list = bb_graph_list
        self.device = device
        self.score_normalizer = score_normalizer
        self.deserializer = deserializer
        self.top_k = top_k

    def run(
        self,
        initial_sequences: list,
        mol_scorers: list[Callable],
        conditioning: torch.TensorType | None = None,
        conditioning_mask: torch.TensorType | None = None,
    ):
        """Run the pipeline to generate completions from initial sequences."""
        # set up initial data structure
        data = get_initial_data(
            self.tkn_lib_collection, initial_sequences, mol_scorers, self.bb_graph_list
        )
        data = data.to_device(self.device)
        batch_size = len(data)

        # run the model
        full_completions = inference_strategies.BatchCompletedSequences(
            token_lib_collection=self.tkn_lib_collection, score_normalizer=self.score_normalizer
        )
        current_sequences, full_completions = self.generator.generate(
            data, full_completions, conditioning, conditioning_mask
        )

        # get the top k completions and convert to a list of list of dicts
        completed_top_k_sequences = full_completions.get_top_k(self.top_k)
        out = []
        for i in range(batch_size):
            completions = []
            for seq in completed_top_k_sequences[i]:
                seq = seq.to_str_form(full_completions.token_lib_collection)
                completions.append(
                    {
                        "sequence": seq.sequence,
                        "score": seq.score,
                    }
                )
            out.append(completions)

        return out


def get_initial_data(
    tkn_lib_collection: vocab.TokenLibraryCollection,
    initial_sequences: list,
    mol_scorers: list[Callable],
    bb_graph_list: list[pyg_data.Data],
):
    """Turns list of prefixes (or otherwise started sequences) into a CurrentSequences object for completion."""

    initial_batch = dataset.RxnNetTaskBatch.from_tokens(
        input_sequences=initial_sequences,
        input_mol_scores=mol_scorers,
        tkn_lib_collection=tkn_lib_collection,
        bb_graph_list=bb_graph_list,
    )

    initial_batch = dataset.CurrentSequences(
        **initial_batch.__dict__,
        example_indices=torch.arange(len(initial_sequences)),
        mol_scorer={i: ms for i, ms in enumerate(mol_scorers)},
    )
    return initial_batch


class GeneratorMethods(enum.Enum):
    SAMPLE = "sample"
    BEAM_SEARCH = "beam_search"


def get_generator(
    model: synthesis_decoder.SynthesisDecoderModel,
    token_action_handler: token_action_handler.TokenActionHandler,
    method: GeneratorMethods,
    **kwargs,
):

    wrapped_model = inference_strategies.ModelActorWrapper(
        model=model,
        token_action_handler=token_action_handler,
        scores_as_probs=False,  # keep them as logits
    )
    warper = kwargs.get("warper", None)
    if warper is not None:
        if warper == "top_k":
            warper = inference_strategies.ComposePDistributionWarper(
                inference_strategies.TopK(k=10),
                inference_strategies.LogitsToLogProbs(),
            )
        else:
            warper = inference_strategies.LogitsToLogProbs()
    else:
        warper = inference_strategies.LogitsToLogProbs()
    if method is GeneratorMethods.SAMPLE:
        strategy = inference_strategies.SamplingInferenceStrategy(
            token_action_handler=token_action_handler,
            one_step_actor=wrapped_model,
            warper=warper,
            stopping_criteria=inference_strategies.StoppingCriteria.UNTIL_STOP,
            number_samples_per_example=kwargs.get("number_samples_per_example", 1),
            max_num_steps=kwargs.get("max_num_steps", 100),
        )
    elif method is GeneratorMethods.BEAM_SEARCH:
        strategy = inference_strategies.BeamSearchInferenceStrategy(
            token_action_handler=token_action_handler,
            one_step_actor=wrapped_model,
            warper=inference_strategies.LogitsToLogProbs(),
            stopping_criteria=inference_strategies.StoppingCriteria.UNTIL_STOP,
            beam_width=kwargs.get("beam_width", 5),
            max_num_steps=kwargs.get("max_num_steps", 100),
        )
    else:
        raise NotImplementedError(f"Generator method {method} not implemented")

    return strategy


class ScoreNormalizer(enum.Enum):
    """Methods for normalizing scores."""

    NONE = "none"
    LENGTH = "length"


def get_score_normalizer(normalizer: ScoreNormalizer, **kwargs):
    if normalizer is ScoreNormalizer.NONE:
        return None
    elif normalizer is ScoreNormalizer.LENGTH:

        def _by_length(sequence: tuple[int, ...], score: float) -> float:
            seq_len = max(1, len(sequence))
            length_penalty = float(seq_len) ** 0.5
            return score / length_penalty

        return _by_length
    else:
        raise NotImplementedError(f"Score normalizer {normalizer} not implemented")
