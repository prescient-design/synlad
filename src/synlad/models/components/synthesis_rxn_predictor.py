import logging
import operator
import pathlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar
from urllib.parse import urlparse, urlunparse

import cachetools
import numpy as np
import requests
from pydantic import BaseModel

from synlad.data.components import synthesis_reaction_network_graph as reaction_network_graph
from synlad.utils import synthesis_chem_utils as chem_utils
from synlad.utils.synthesis_utils import ManyToManyMapping, compute_file_sha256

T = TypeVar("T")

# Constants
PREDICTION_REQUEST_TIMEOUT_SECONDS = 180  # 3 minutes


class RxnLMConnectionError(Exception):
    """Exception raised when unable to communicate with the prediction service."""

    def __init__(self, server_url: str, original_error: Exception = None):
        self.server_url = server_url
        self.original_error = original_error
        message = f"Failed to communicate with prediction service at {server_url}"
        if original_error:
            message += f": {str(original_error)}"
        super().__init__(message)


# we don't have an installable class for rxn-lm predictor API, so we will
# just hardcode the input and output types here


class PredictInput(BaseModel):
    # match the input type of the pipeline so with duck typing we can use this
    smiles: str
    direction: Literal["forward", "backward"]


class IndividualPrediction(BaseModel):
    smiles: str
    score: float


class PredictOutput(BaseModel):
    smiles: str
    direction: Literal["forward", "backward"]
    predictions: list[IndividualPrediction]


class PredictRequest(BaseModel):
    task_metadata: dict
    inputs: list[PredictInput]


class PredictResponse(BaseModel):
    outputs: list[PredictOutput]


@dataclass(frozen=True)  # frozen so that it can be used as a key in a dict/set
class RxnLMPredictionInput:
    canonical_smiles_set: frozenset
    direction: Literal["forward", "backward"]

    @classmethod
    def from_product_smi(cls, retro_smi: str, skip_canonicalization: bool = False):
        canon_smi = retro_smi if skip_canonicalization else chem_utils.canonicalize(retro_smi)
        return cls(canonical_smiles_set=frozenset({canon_smi}), direction="backward")

    @classmethod
    def from_reactant_smis(cls, reactant_smis: list[str], skip_canonicalization: bool = False):
        # note currently removes duplicate reactants.
        canon_ = (lambda x: x) if skip_canonicalization else chem_utils.canonicalize
        canon_smis = frozenset({canon_(s) for s in reactant_smis})
        return cls(canonical_smiles_set=canon_smis, direction="forward")

    def to_predict_input(self) -> PredictInput:
        return PredictInput(
            smiles=".".join(sorted(list(self.canonical_smiles_set))), direction=self.direction
        )


def _retry_n_times(func: Callable[[], T], n_retries: int = 3, logger=None) -> T:  # noqa: UP047
    """Retry a function n times, catching exceptions and retrying.

    Args:
        func: Function to retry
        n_retries: Number of retry attempts (default: 3)
        logger: Optional custom logger (default: None, uses built-in logging)

    Returns:
        The return value of the function if successful

    Raises:
        Exception: The last exception raised if all retries fail
    """
    last_exception = None
    for attempt in range(n_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            log_message = f"Attempt {attempt + 1}/{n_retries} failed: {str(e)}"
            if logger:
                logger.warning(log_message)
            else:
                logging.warning(log_message)
            if attempt < n_retries - 1:
                # Wait a bit before retrying (with exponential backoff)
                time.sleep(2**attempt)

    # If we got here, all retries failed
    raise last_exception


class ReactionPredictor(Protocol):
    """Protocol for reaction prediction models.

    This protocol defines the common interface for reaction predictors that can
    predict reaction products from reactants (forward direction) or reactants
    from products (retrosynthesis/backward direction).

    Implementations can be initialized in different ways but must provide
    the predict method with the specified signature.
    """

    def predict(self, inputs: list[RxnLMPredictionInput]) -> list[Iterable[set[str]]]:
        """Predict reaction outcomes for the given inputs.

        Args:
            inputs: List of prediction inputs containing reactants/products and direction

        Returns:
            List of sets of predicted molecule SMILES strings, one set per input
        """
        ...


class SetOnFirstPredictorWithAttr:
    def set_attr_on_relevant_predictor(self, attr_name: str, value: Any):
        """For a wrapped predictor, set the attribute on the first predictor that has it."""
        return self._set_attr(self, attr_name, value)

    @classmethod
    def _set_attr(cls, obj: "ReactionPredictor", attr_name: str, value: Any):
        if not hasattr(obj, attr_name):
            if hasattr(obj, "predictor"):
                cls._set_attr(obj.predictor, attr_name, value)
            else:
                raise AttributeError(
                    f"No predictor attribute found on {obj} and no predictor attribute on predictor {obj.predictor}"
                )
        else:
            setattr(obj, attr_name, value)
        return obj


class CachedReactionPredictor(ReactionPredictor, SetOnFirstPredictorWithAttr):
    """Wrapper that adds caching to any ReactionPredictor implementation.

    This decorator implements the ReactionPredictor protocol and can wrap
    any predictor to add LRU caching functionality.
    """

    def __init__(self, predictor: ReactionPredictor, cache_size: int = 500_000):
        """Initialize the cached wrapper.

        Args:
            predictor: The underlying predictor to wrap
            cache_size: Maximum number of predictions to cache (default: 500,000)
        """
        self.predictor = predictor
        self.cache = cachetools.LRUCache(maxsize=cache_size)

    def predict(self, inputs: list[RxnLMPredictionInput]) -> list[Iterable[set[str]]]:
        """Predict with caching support."""
        output_predictions = {}  # maps input index to set of predicted smiles
        to_predict: list[RxnLMPredictionInput] = []  # inputs not in cache
        predindx_to_input_indx = {}  # maps prediction index to input index (for returning results in original order)

        # Check cache first and gather inputs that need prediction
        for i, input_item in enumerate(inputs):
            if input_item in self.cache:
                # If already in cache, add directly to output
                output_predictions[i] = self.cache[input_item]
            else:
                # Add to list for batch prediction
                predindx_to_input_indx[len(to_predict)] = i
                to_predict.append(input_item)

        # If there are inputs that need prediction, delegate to wrapped predictor
        if to_predict:
            predictions = self.predictor.predict(to_predict)

            # Store results in cache and add to output
            for i, prediction_set in enumerate(predictions):
                # Store in cache
                original_input = to_predict[i]
                self.cache[original_input] = prediction_set

                # Add to output in original order
                original_index = predindx_to_input_indx[i]
                output_predictions[original_index] = prediction_set

        # Return results in the original order
        return [output_predictions[i] for i in range(len(inputs))]


class SingleOutputReactionPredictor(ReactionPredictor, SetOnFirstPredictorWithAttr):
    """Wrapper that ensures the output is just a (randomly selected) single prediction for each input."""

    def __init__(self, predictor: ReactionPredictor, rng: np.random.RandomState):
        self.predictor = predictor
        self.rng = rng

    def predict(self, inputs: list[RxnLMPredictionInput]) -> list[Iterable[set[str]]]:
        out = self.predictor.predict(inputs)
        new_out = []
        for o in out:
            new_out.append([self.rng.choice(list(o))])
        return new_out


class FirstOutputReactionPredictor(ReactionPredictor, SetOnFirstPredictorWithAttr):
    """Wrapper that ensures the output is just the first prediction for each input."""

    def __init__(self, predictor: ReactionPredictor):
        self.predictor = predictor

    def predict(self, inputs: list[RxnLMPredictionInput]) -> list[Iterable[set[str]]]:
        out = self.predictor.predict(inputs)
        return [[next(iter(o))] for o in out]


class RxnLM(ReactionPredictor):
    """Calls the RXN-LM predictor API to get predictions for a list of reactants or products."""

    def __init__(
        self,
        prediction_url: str,
        logger=None,
        if_none_return_input: bool = True,
        skip_canonicalization: bool = False,
        model_details_url: str | None = None,
    ):
        self.prediction_url = prediction_url
        self.logger = logger
        self._if_none_return_input = if_none_return_input
        self._skip_canonicalization = skip_canonicalization

        # If no model details url is provided, try and infer one from the prediction url.
        if model_details_url is None:
            parsed = urlparse(prediction_url)
            model_details_url = urlunparse(
                parsed._replace(path="model_info", params="", query="", fragment="")
            )
        self.model_details_url = model_details_url

        # & then query model details endpoint and log the details.
        try:
            response = requests.get(self.model_details_url, timeout=10)
            response.raise_for_status()
            model_details = response.json()
            if self.logger:
                self.logger.info(f"Model details from {self.model_details_url}: {model_details}")
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to fetch model details from {self.model_details_url}: {str(e)}"
                )

    def predict(self, inputs: list[RxnLMPredictionInput]) -> list[Iterable[set[str]]]:
        """Runs predictions on all the inputs. for each input returns a set of all the canonicalized molecule strings
        that were predicted.

        At moment we only return the top-1 prediction.
        """
        if not inputs:
            return []

        # Convert inputs to prediction request format
        predict_inputs = [item.to_predict_input() for item in inputs]

        # Create prediction request
        request = PredictRequest(task_metadata={}, inputs=predict_inputs)

        # Will send batch to prediction service with retries
        # Note: This allows the server to do batching as it likely better knows how to optimize batches
        def make_prediction_request():
            try:
                response = requests.post(
                    self.prediction_url,
                    data=request.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                    timeout=PREDICTION_REQUEST_TIMEOUT_SECONDS,
                )
                return PredictResponse.model_validate(response.json())
            except Exception as e:
                raise RxnLMConnectionError(self.prediction_url, e) from e

        # Retry the request up to 3 times
        predict_response = _retry_n_times(make_prediction_request, n_retries=3, logger=self.logger)

        # Process responses
        results = []
        for i, output in enumerate(predict_response.outputs):
            outputs_for_one_input = []
            for prediction in output.predictions:
                prediction_set = set(prediction.smiles.split("."))
                if not self._skip_canonicalization:
                    prediction_set = {chem_utils.canonicalize(s) for s in prediction_set}
                    prediction_set = prediction_set - {None}

                if self._if_none_return_input and len(prediction_set) == 0:
                    prediction_set = inputs[i].canonical_smiles_set

                outputs_for_one_input.append(prediction_set)

            results.append(outputs_for_one_input)

        return results


class RxnDataLookup(ReactionPredictor):
    """Reaction predictor that uses a dictionary lookup from a set of ground truth reactions."""

    def __init__(
        self,
        reaction_smiles: list[str],
        fallback_molecules: frozenset = frozenset({"CO"}),
        logger=None,
        canonicalize: bool = True,
        reaction_selector: Callable | None = None,
    ):
        """
        Args:
            reaction_smiles: List of reaction SMILES strings in format "reactants>>products"
            fallback_molecules: Molecules to return if no mapping found (default: {"CO"})
            (we do this so that can see these cases easily, but could instead in future just return the input molecules)
            logger: Optional logger for error reporting
            canonicalize: Whether to canonicalize the SMILES strings when using this class.
            product_selector: Optional function to select which products to return when multiple exist.
                             Takes set of tuples (frozenset_reactants, frozenset_products) where each tuple
                             contains single product sets, and returns filtered set of selected tuples. Should only
                             be one product in frozenset.
                             If None, returns all reactions (default behavior).
        """
        self.fallback_molecules = fallback_molecules
        self.logger = logger
        self._reactants_to_products = ManyToManyMapping()
        self._load_reactions(reaction_smiles, canonicalize=canonicalize)
        self._canonicalize = canonicalize
        self.product_selector = reaction_selector

    @classmethod
    def from_rxnnet_pickle_file(cls, rxnnet_pickle_file: pathlib.Path, **kwargs):
        "create a reaction predictor from the rxnnet"

        if (logger := kwargs.get("logger")) is not None:
            file_hash = compute_file_sha256(rxnnet_pickle_file)
            logger.info(f"Loading reaction network from: {rxnnet_pickle_file}")
            logger.info(f"File hash (SHA256): {file_hash}")

        rxnnet = reaction_network_graph.ReactionNetwork.load_from_file(rxnnet_pickle_file)
        rxns = [repr(el) for el in rxnnet.edges]
        return cls(rxns, **kwargs)

    @staticmethod
    def _parse_and_canonicalize_smiles(smiles_string: str, canonicalize: bool = True) -> frozenset:
        """Parse dot-separated SMILES and canonicalize them into a frozenset."""
        smiles_list = smiles_string.split(".")
        canonicalized = []

        for smi in smiles_list:
            canon_smi = chem_utils.canonicalize(smi.strip()) if canonicalize else smi.strip()
            if canon_smi:  # Only add if canonicalization succeeded
                canonicalized.append(canon_smi)

        return frozenset(canonicalized)

    def _load_reactions(self, reaction_smiles: list[str], canonicalize: bool = True):
        """Parse reaction SMILES and populate the mapping."""
        failed_count = 0

        for i, rxn_smi in enumerate(reaction_smiles):
            if ">>" not in rxn_smi:
                failed_count += 1
                if self.logger:
                    self.logger.warning(
                        f"Line {i + 1}: No '>>' found in reaction SMILES: {rxn_smi}"
                    )
                continue

            reactants_str, products_str = rxn_smi.split(">>", 1)  # Split on first occurrence only

            # Parse and canonicalize both sides
            reactants_set = self._parse_and_canonicalize_smiles(reactants_str, canonicalize)
            products_set = self._parse_and_canonicalize_smiles(products_str, canonicalize)

            if reactants_set and products_set:
                # Add bidirectional mapping
                self._reactants_to_products.add_single_relationship(reactants_set, products_set)
            else:
                failed_count += 1
                if self.logger:
                    self.logger.warning(f"Line {i + 1}: Failed to canonicalize reaction: {rxn_smi}")

        if self.logger:
            total_reactions = len(reaction_smiles)
            successful_reactions = total_reactions - failed_count
            self.logger.info(
                f"Loaded {successful_reactions}/{total_reactions} reactions successfully"
            )

    def predict(self, inputs: list[RxnLMPredictionInput]) -> list[Iterable[set[str]]]:
        """Predict reaction outcomes using dictionary lookup.

        Args:
            inputs: List of prediction inputs containing reactants/products and direction

        Returns:
            List of sets of predicted molecule SMILES strings, one set per input
        """
        if not inputs:
            return []

        # Create inversed mapping once for efficiency
        products_to_reactants = self._reactants_to_products.inversed

        results = []
        for input_item in inputs:
            # Select appropriate mapping based on direction
            if input_item.direction == "forward":
                mapping = self._reactants_to_products
            else:  # backward
                mapping = products_to_reactants

            # Look up prediction using shared logic
            prediction_result = self._lookup_single_prediction(
                input_item.canonical_smiles_set, mapping, input_item.direction
            )

            results.append(prediction_result)

        return results

    def _lookup_single_prediction(
        self, query_set: frozenset, mapping: ManyToManyMapping, direction: str
    ) -> set[frozenset]:
        """Look up prediction in mapping and apply product selection logic. If not there, return fallback."""
        try:
            result_sets = mapping[query_set]
        except KeyError:
            result_sets = frozenset()

        # Apply product selector if provided
        if self.product_selector is not None:
            # Results sets represent product sets if forward, and reactants sets if backward.
            slice_dir = 1 if direction == "forward" else -1
            reaction_tuples = {(query_set, o)[::slice_dir] for o in result_sets}

            # Get filtered tuples from selector -- this always expects (frozenset(reactants), frozenset(products))
            selected_tuples = self.product_selector(reaction_tuples)

            # select out products if forward, and reactants if backward.
            selector = operator.itemgetter(1 if direction == "forward" else 0)
            result_sets = {selector(t) for t in selected_tuples}  # a set of frozensets

        if len(result_sets) > 0:
            return result_sets
        else:
            # No mapping found, return fallback
            return {
                self.fallback_molecules,
            }
