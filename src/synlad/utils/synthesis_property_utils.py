"""Property/scoring utilities for synthesis pathway tasks.

Mirrors `everydog.property_tasks.property_utils` so that pickled `RxnNetTask`s
referencing those classes can be loaded without `everydog` installed.
"""

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


class TanimotoSimScorer:
    def __init__(self, base_smi, **kwargs):
        self.mol1 = Chem.MolFromSmiles(base_smi)
        if self.mol1 is None:
            raise ValueError(f"could not create mol from {base_smi}")

        self._kwargs = kwargs
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(**kwargs)
        self.base_fp = self.fpgen.GetFingerprint(self.mol1)

    def __call__(self, smi):
        mol2 = Chem.MolFromSmiles(smi)
        if mol2 is None:
            raise ValueError(f"could not create mol from {smi}")
        fp2 = self.fpgen.GetFingerprint(mol2)
        return Chem.DataStructs.TanimotoSimilarity(self.base_fp, fp2)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["fpgen"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(**self._kwargs)
