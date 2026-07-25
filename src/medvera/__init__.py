"""MedVERA: protocol-aware trustworthiness verification for longitudinal EHRs."""

from .consensus_refinement import ConsensusEvidenceRefiner
from .evidence_construction import ProtocolAwareEvidenceConstructor
from .exact_set_scoring import ExactSetScorer
from .pipeline import MedVERAPipeline
from .semantic_assessment import SemanticTrustRiskAssessor
from .training import TrainableMedVERA, TrainingConfig

__all__ = [
    "ConsensusEvidenceRefiner",
    "ExactSetScorer",
    "MedVERAPipeline",
    "ProtocolAwareEvidenceConstructor",
    "SemanticTrustRiskAssessor",
    "TrainableMedVERA",
    "TrainingConfig",
]

__version__ = "1.1.0"
