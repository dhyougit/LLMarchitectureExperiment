"""
LLM Architecture Experimentation Framework
"""
from .config import ExperimentConfig, ExperimentResults
from .model import CustomGPT2Model
from .trainer import ArchitectureExperiment
from .metrics import MetricsCalculator, MetricsTracker
from .analysis import compare_experiments, plot_training_curves, generate_summary_report

__version__ = "0.1.0"

__all__ = [
    'ExperimentConfig',
    'ExperimentResults',
    'CustomGPT2Model',
    'ArchitectureExperiment',
    'MetricsCalculator',
    'MetricsTracker',
    'compare_experiments',
    'plot_training_curves',
    'generate_summary_report',
]
