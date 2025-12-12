#!/usr/bin/env python3
"""
MQIASE™ xAI-Ready Live Demo v3.0
Advanced eXplainable Artificial Intelligence Framework
Complete Implementation with Real-time Visualization

Author: Amjad Al-Baaj
Contact: ba.aj@hotmail.com
Date: 2025-12-12
License: MIT

Description:
    This module provides a complete implementation of MQIASE™ (Multi-Quality Interpretability
    and Assurance System for eXplainability), featuring three core evaluation metrics:
    - ITQAN Metric: Integrity and Quality Assessment
    - MIZAN Metric: Balance and Harmony Analysis
    - TAFAKKUR Metric: Reflection and Deep Analysis
    
    All metrics support multiple fidelity levels for progressive analysis with live
    visualization using matplotlib for real-time monitoring and interpretation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FidelityLevel(Enum):
    """Fidelity levels for metric computation."""
    BASIC = 1
    STANDARD = 2
    ADVANCED = 3
    EXPERT = 4
    ELITE = 5


@dataclass
class MetricResult:
    """Data class for metric computation results."""
    name: str
    score: float
    fidelity_level: FidelityLevel
    components: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.95
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __str__(self) -> str:
        return (f"{self.name} Score: {self.score:.4f} | "
                f"Fidelity: {self.fidelity_level.name} | "
                f"Confidence: {self.confidence:.2%}")


@dataclass
class VisualizationConfig:
    """Configuration for live visualization."""
    update_interval: int = 100  # milliseconds
    max_data_points: int = 100
    show_grid: bool = True
    show_legend: bool = True
    figure_size: Tuple[int, int] = (16, 10)
    dpi: int = 100


class ITQANMetric:
    """
    ITQAN Metric: Integrity and Quality Assessment
    
    Evaluates the integrity, quality, and reliability of model interpretations.
    Scores range from 0 to 1, where 1 represents perfect integrity.
    """
    
    def __init__(self, name: str = "ITQAN Metric"):
        self.name = name
        self.history: List[MetricResult] = []
    
    def compute(self, 
                predictions: np.ndarray,
                ground_truth: np.ndarray,
                fidelity: FidelityLevel = FidelityLevel.STANDARD) -> MetricResult:
        """
        Compute ITQAN metric for given predictions and ground truth.
        
        Args:
            predictions: Model predictions array
            ground_truth: Ground truth labels array
            fidelity: Fidelity level for computation
            
        Returns:
            MetricResult object with ITQAN score and components
        """
        components = {}
        
        # Accuracy component
        accuracy = np.mean(predictions == ground_truth)
        components['accuracy'] = accuracy
        
        # Consistency component
        if len(self.history) > 0:
            previous_score = self.history[-1].score
            consistency = 1.0 - abs(accuracy - previous_score)
            components['consistency'] = consistency
        else:
            components['consistency'] = 1.0
        
        # Fidelity-dependent components
        if fidelity == FidelityLevel.BASIC:
            score = accuracy
            
        elif fidelity == FidelityLevel.STANDARD:
            reliability = np.std(predictions) / (np.mean(np.abs(predictions)) + 1e-8)
            components['reliability'] = 1.0 / (1.0 + reliability)
            score = (accuracy + components['consistency'] + components['reliability']) / 3.0
            
        elif fidelity == FidelityLevel.ADVANCED:
            # Add confidence analysis
            confidence_score = np.mean(np.max(
                np.column_stack([predictions, 1.0 - predictions]), axis=1
            )) if len(predictions.shape) > 1 or np.max(predictions) <= 1.0 else 0.8
            components['confidence_score'] = confidence_score
            
            reliability = np.std(predictions) / (np.mean(np.abs(predictions)) + 1e-8)
            components['reliability'] = 1.0 / (1.0 + reliability)
            
            score = (0.3 * accuracy + 0.25 * components['consistency'] + 
                    0.25 * components['reliability'] + 0.2 * confidence_score)
            
        elif fidelity == FidelityLevel.EXPERT:
            # Comprehensive analysis
            confidence_score = np.mean(np.max(
                np.column_stack([predictions, 1.0 - predictions]), axis=1
            )) if len(predictions.shape) > 1 or np.max(predictions) <= 1.0 else 0.8
            components['confidence_score'] = confidence_score
            
            reliability = np.std(predictions) / (np.mean(np.abs(predictions)) + 1e-8)
            components['reliability'] = 1.0 / (1.0 + reliability)
            
            # Variance analysis
            variance_stability = 1.0 - np.clip(np.var(predictions), 0, 1)
            components['variance_stability'] = variance_stability
            
            # Bias analysis
            bias = np.mean(predictions) - np.mean(ground_truth)
            components['bias'] = 1.0 - np.clip(abs(bias), 0, 1)
            
            score = (0.25 * accuracy + 0.2 * components['consistency'] + 
                    0.2 * components['reliability'] + 0.15 * confidence_score +
                    0.1 * variance_stability + 0.1 * components['bias'])
            
        else:  # ELITE
            # Maximum depth analysis
            confidence_score = np.mean(np.max(
                np.column_stack([predictions, 1.0 - predictions]), axis=1
            )) if len(predictions.shape) > 1 or np.max(predictions) <= 1.0 else 0.8
            components['confidence_score'] = confidence_score
            
            reliability = np.std(predictions) / (np.mean(np.abs(predictions)) + 1e-8)
            components['reliability'] = 1.0 / (1.0 + reliability)
            
            variance_stability = 1.0 - np.clip(np.var(predictions), 0, 1)
            components['variance_stability'] = variance_stability
            
            bias = np.mean(predictions) - np.mean(ground_truth)
            components['bias'] = 1.0 - np.clip(abs(bias), 0, 1)
            
            # Additional elite components
            robustness = np.mean(1.0 - np.abs(predictions - ground_truth))
            components['robustness'] = robustness
            
            calibration = np.std(predictions - ground_truth)
            components['calibration'] = 1.0 - np.clip(calibration, 0, 1)
            
            score = (0.2 * accuracy + 0.15 * components['consistency'] + 
                    0.15 * components['reliability'] + 0.12 * confidence_score +
                    0.12 * variance_stability + 0.1 * components['bias'] +
                    0.1 * robustness + 0.06 * components['calibration'])
        
        # Clip score to valid range
        score = np.clip(score, 0.0, 1.0)
        
        result = MetricResult(
            name=self.name,
            score=score,
            fidelity_level=fidelity,
            components=components,
            confidence=0.95 if fidelity.value >= 3 else 0.85
        )
        
        self.history.append(result)
        logger.info(f"ITQAN Metric computed: {result}")
        
        return result


class MIZANMetric:
    """
    MIZAN Metric: Balance and Harmony Analysis
    
    Evaluates the balance, harmony, and equilibrium of model behavior.
    Scores range from 0 to 1, where 1 represents perfect balance.
    """
    
    def __init__(self, name: str = "MIZAN Metric"):
        self.name = name
        self.history: List[MetricResult] = []
    
    def compute(self,
                predictions: np.ndarray,
                class_weights: Optional[np.ndarray] = None,
                fidelity: FidelityLevel = FidelityLevel.STANDARD) -> MetricResult:
        """
        Compute MIZAN metric for class balance and distribution harmony.
        
        Args:
            predictions: Model predictions array
            class_weights: Optional class weights for weighted analysis
            fidelity: Fidelity level for computation
            
        Returns:
            MetricResult object with MIZAN score and components
        """
        components = {}
        
        # Calculate class distribution
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = counts / len(predictions)
        components['distribution_uniformity'] = 1.0 - np.std(distribution)
        
        # Entropy-based balance
        entropy = -np.sum(distribution * np.log(distribution + 1e-10))
        max_entropy = np.log(len(unique))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        components['entropy_balance'] = normalized_entropy
        
        # Fidelity-dependent components
        if fidelity == FidelityLevel.BASIC:
            score = components['entropy_balance']
            
        elif fidelity == FidelityLevel.STANDARD:
            # Add distribution harmony
            components['harmony'] = 1.0 - np.std(counts / len(predictions))
            score = (components['entropy_balance'] + components['harmony']) / 2.0
            
        elif fidelity == FidelityLevel.ADVANCED:
            # Add asymmetry analysis
            components['harmony'] = 1.0 - np.std(counts / len(predictions))
            asymmetry = np.sum(np.abs(distribution - (1.0 / len(unique))))
            components['asymmetry'] = 1.0 - np.clip(asymmetry, 0, 1)
            
            score = (0.35 * components['entropy_balance'] + 
                    0.35 * components['harmony'] + 
                    0.3 * components['asymmetry'])
            
        elif fidelity == FidelityLevel.EXPERT:
            # Comprehensive balance analysis
            components['harmony'] = 1.0 - np.std(counts / len(predictions))
            asymmetry = np.sum(np.abs(distribution - (1.0 / len(unique))))
            components['asymmetry'] = 1.0 - np.clip(asymmetry, 0, 1)
            
            # Gini coefficient for inequality
            sorted_dist = np.sort(distribution)
            gini = (2.0 * np.sum((np.arange(1, len(sorted_dist) + 1)) * sorted_dist) / 
                   (len(sorted_dist) * np.sum(sorted_dist))) - (len(sorted_dist) + 1) / len(sorted_dist)
            components['gini_coefficient'] = 1.0 - gini
            
            # Equilibrium score
            if class_weights is not None:
                weighted_balance = np.std(distribution * class_weights)
                components['weighted_balance'] = 1.0 - np.clip(weighted_balance, 0, 1)
            else:
                components['weighted_balance'] = 0.5
            
            score = (0.3 * components['entropy_balance'] + 
                    0.25 * components['harmony'] + 
                    0.2 * components['asymmetry'] + 
                    0.15 * components['gini_coefficient'] + 
                    0.1 * components['weighted_balance'])
            
        else:  # ELITE
            # Maximum depth balance analysis
            components['harmony'] = 1.0 - np.std(counts / len(predictions))
            asymmetry = np.sum(np.abs(distribution - (1.0 / len(unique))))
            components['asymmetry'] = 1.0 - np.clip(asymmetry, 0, 1)
            
            sorted_dist = np.sort(distribution)
            gini = (2.0 * np.sum((np.arange(1, len(sorted_dist) + 1)) * sorted_dist) / 
                   (len(sorted_dist) * np.sum(sorted_dist))) - (len(sorted_dist) + 1) / len(sorted_dist)
            components['gini_coefficient'] = 1.0 - gini
            
            if class_weights is not None:
                weighted_balance = np.std(distribution * class_weights)
                components['weighted_balance'] = 1.0 - np.clip(weighted_balance, 0, 1)
            else:
                components['weighted_balance'] = 0.5
            
            # Additional elite components
            kurtosis = np.sum((distribution - np.mean(distribution))**4) / (len(distribution) * np.std(distribution)**4 + 1e-10)
            components['distribution_kurtosis'] = 1.0 - np.clip(kurtosis / 10.0, 0, 1)
            
            # Stability metric
            if len(self.history) > 0:
                previous_distribution = self.history[-1].components.get('distribution_uniformity', 0.5)
                stability = 1.0 - abs(components['distribution_uniformity'] - previous_distribution)
                components['stability'] = stability
            else:
                components['stability'] = 1.0
            
            score = (0.25 * components['entropy_balance'] + 
                    0.2 * components['harmony'] + 
                    0.15 * components['asymmetry'] + 
                    0.15 * components['gini_coefficient'] + 
                    0.1 * components['weighted_balance'] + 
                    0.1 * components['distribution_kurtosis'] + 
                    0.05 * components['stability'])
        
        # Clip score to valid range
        score = np.clip(score, 0.0, 1.0)
        
        result = MetricResult(
            name=self.name,
            score=score,
            fidelity_level=fidelity,
            components=components,
            confidence=0.95 if fidelity.value >= 3 else 0.85
        )
        
        self.history.append(result)
        logger.info(f"MIZAN Metric computed: {result}")
        
        return result


class TAFAKKURMetric:
    """
    TAFAKKUR Metric: Reflection and Deep Analysis
    
    Evaluates the depth, insight quality, and analytical value of model interpretations.
    Scores range from 0 to 1, where 1 represents maximum analytical depth.
    """
    
    def __init__(self, name: str = "TAFAKKUR Metric"):
        self.name = name
        self.history: List[MetricResult] = []
    
    def compute(self,
                feature_importance: Union[np.ndarray, Dict[str, float]],
                explanation_quality: Optional[np.ndarray] = None,
                fidelity: FidelityLevel = FidelityLevel.STANDARD) -> MetricResult:
        """
        Compute TAFAKKUR metric for analytical depth and insight quality.
        
        Args:
            feature_importance: Feature importance scores or dictionary
            explanation_quality: Optional explanation quality scores
            fidelity: Fidelity level for computation
            
        Returns:
            MetricResult object with TAFAKKUR score and components
        """
        components = {}
        
        # Convert feature importance to array if needed
        if isinstance(feature_importance, dict):
            feature_importance = np.array(list(feature_importance.values()))
        
        feature_importance = np.array(feature_importance).flatten()
        
        # Analytical depth
        depth = np.sum(feature_importance) / len(feature_importance) if len(feature_importance) > 0 else 0
        components['analytical_depth'] = depth
        
        # Concentration/Focus
        concentration = np.max(feature_importance) - np.min(feature_importance)
        components['focus_concentration'] = np.clip(concentration, 0, 1)
        
        # Fidelity-dependent components
        if fidelity == FidelityLevel.BASIC:
            score = components['analytical_depth']
            
        elif fidelity == FidelityLevel.STANDARD:
            # Add insight diversity
            normalized_importance = feature_importance / (np.sum(feature_importance) + 1e-10)
            diversity = -np.sum(normalized_importance * np.log(normalized_importance + 1e-10))
            components['insight_diversity'] = diversity / np.log(max(len(feature_importance), 2))
            
            score = (components['analytical_depth'] + components['focus_concentration'] + 
                    components['insight_diversity']) / 3.0
            
        elif fidelity == FidelityLevel.ADVANCED:
            # Enhanced insight analysis
            normalized_importance = feature_importance / (np.sum(feature_importance) + 1e-10)
            diversity = -np.sum(normalized_importance * np.log(normalized_importance + 1e-10))
            components['insight_diversity'] = diversity / np.log(max(len(feature_importance), 2))
            
            # Coverage analysis
            coverage = np.sum(feature_importance > np.mean(feature_importance)) / len(feature_importance)
            components['coverage'] = coverage
            
            if explanation_quality is not None:
                components['explanation_quality'] = np.mean(explanation_quality)
                score = (0.3 * components['analytical_depth'] + 
                        0.25 * components['insight_diversity'] + 
                        0.2 * components['focus_concentration'] + 
                        0.15 * components['coverage'] + 
                        0.1 * components['explanation_quality'])
            else:
                score = (0.35 * components['analytical_depth'] + 
                        0.3 * components['insight_diversity'] + 
                        0.2 * components['focus_concentration'] + 
                        0.15 * components['coverage'])
            
        elif fidelity == FidelityLevel.EXPERT:
            # Comprehensive analytical evaluation
            normalized_importance = feature_importance / (np.sum(feature_importance) + 1e-10)
            diversity = -np.sum(normalized_importance * np.log(normalized_importance + 1e-10))
            components['insight_diversity'] = diversity / np.log(max(len(feature_importance), 2))
            
            coverage = np.sum(feature_importance > np.mean(feature_importance)) / len(feature_importance)
            components['coverage'] = coverage
            
            # Redundancy analysis
            if len(feature_importance) > 1:
                correlation_sum = np.sum(np.corrcoef(feature_importance.reshape(-1, 1))[0, 1:])
                redundancy = 1.0 - np.clip(correlation_sum / (len(feature_importance) - 1), 0, 1)
            else:
                redundancy = 1.0
            components['redundancy_reduction'] = redundancy
            
            if explanation_quality is not None:
                components['explanation_quality'] = np.mean(explanation_quality)
                quality_weight = 0.1
            else:
                quality_weight = 0.0
            
            # Coherence score
            coherence = 1.0 - np.std(feature_importance / (np.mean(feature_importance) + 1e-10))
            components['coherence'] = np.clip(coherence, 0, 1)
            
            score = (0.25 * components['analytical_depth'] + 
                    0.25 * components['insight_diversity'] + 
                    0.15 * components['focus_concentration'] + 
                    0.15 * components['coverage'] + 
                    0.1 * components['redundancy_reduction'] + 
                    0.1 * components['coherence'])
            
            if quality_weight > 0:
                score = (0.9 * score + quality_weight * components['explanation_quality'])
            
        else:  # ELITE
            # Maximum depth analytical evaluation
            normalized_importance = feature_importance / (np.sum(feature_importance) + 1e-10)
            diversity = -np.sum(normalized_importance * np.log(normalized_importance + 1e-10))
            components['insight_diversity'] = diversity / np.log(max(len(feature_importance), 2))
            
            coverage = np.sum(feature_importance > np.mean(feature_importance)) / len(feature_importance)
            components['coverage'] = coverage
            
            if len(feature_importance) > 1:
                correlation_sum = np.sum(np.corrcoef(feature_importance.reshape(-1, 1))[0, 1:])
                redundancy = 1.0 - np.clip(correlation_sum / (len(feature_importance) - 1), 0, 1)
            else:
                redundancy = 1.0
            components['redundancy_reduction'] = redundancy
            
            if explanation_quality is not None:
                components['explanation_quality'] = np.mean(explanation_quality)
                quality_weight = 0.1
            else:
                quality_weight = 0.0
            
            coherence = 1.0 - np.std(feature_importance / (np.mean(feature_importance) + 1e-10))
            components['coherence'] = np.clip(coherence, 0, 1)
            
            # Additional elite components
            granularity = np.sum(feature_importance > 0) / len(feature_importance)
            components['granularity'] = granularity
            
            # Temporal consistency (if available)
            if len(self.history) > 0:
                previous_importance = self.history[-1].components.get('analytical_depth', 0.5)
                consistency = 1.0 - abs(components['analytical_depth'] - previous_importance)
                components['temporal_consistency'] = consistency
            else:
                components['temporal_consistency'] = 1.0
            
            score = (0.2 * components['analytical_depth'] + 
                    0.2 * components['insight_diversity'] + 
                    0.12 * components['focus_concentration'] + 
                    0.12 * components['coverage'] + 
                    0.1 * components['redundancy_reduction'] + 
                    0.1 * components['coherence'] + 
                    0.08 * components['granularity'] + 
                    0.08 * components['temporal_consistency'])
            
            if quality_weight > 0:
                score = (0.9 * score + quality_weight * components['explanation_quality'])
        
        # Clip score to valid range
        score = np.clip(score, 0.0, 1.0)
        
        result = MetricResult(
            name=self.name,
            score=score,
            fidelity_level=fidelity,
            components=components,
            confidence=0.95 if fidelity.value >= 3 else 0.85
        )
        
        self.history.append(result)
        logger.info(f"TAFAKKUR Metric computed: {result}")
        
        return result


class MQIAASELiveVisualizer:
    """
    Live visualization system for MQIASE metrics with real-time updates.
    """
    
    def __init__(self,
                 itqan: ITQANMetric,
                 mizan: MIZANMetric,
                 tafakkur: TAFAKKURMetric,
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize the live visualizer.
        
        Args:
            itqan: ITQANMetric instance
            mizan: MIZANMetric instance
            tafakkur: TAFAKKURMetric instance
            config: VisualizationConfig for customization
        """
        self.itqan = itqan
        self.mizan = mizan
        self.tafakkur = tafakkur
        self.config = config or VisualizationConfig()
        
        # Data storage for visualization
        self.timestamps: List[str] = []
        self.itqan_scores: List[float] = []
        self.mizan_scores: List[float] = []
        self.tafakkur_scores: List[float] = []
        
        # Figure setup
        self.fig: Optional[plt.Figure] = None
        self.axes: Dict[str, plt.Axes] = {}
        self.animation_obj: Optional[animation.FuncAnimation] = None
    
    def setup_figure(self) -> None:
        """Set up the matplotlib figure with subplots."""
        self.fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        self.fig.suptitle('MQIASE™ xAI-Ready Live Demo v3.0 - Real-time Visualization',
                         fontsize=16, fontweight='bold')
        
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Main metrics plot
        self.axes['main'] = self.fig.add_subplot(gs[0, :])
        
        # Individual metric plots
        self.axes['itqan'] = self.fig.add_subplot(gs[1, 0])
        self.axes['mizan'] = self.fig.add_subplot(gs[1, 1])
        self.axes['tafakkur'] = self.fig.add_subplot(gs[2, 0])
        
        # Statistics plot
        self.axes['stats'] = self.fig.add_subplot(gs[2, 1])
        
        # Configure axes
        for ax_name in ['main', 'itqan', 'mizan', 'tafakkur']:
            self.axes[ax_name].set_ylim([0, 1])
            self.axes[ax_name].grid(self.config.show_grid, alpha=0.3)
    
    def update_data(self) -> None:
        """Update internal data from metric histories."""
        self.timestamps = [f"t{i}" for i in range(len(self.itqan.history))]
        self.itqan_scores = [r.score for r in self.itqan.history]
        self.mizan_scores = [r.score for r in self.mizan.history]
        self.tafakkur_scores = [r.score for r in self.tafakkur.history]
        
        # Limit data points to configured maximum
        if len(self.timestamps) > self.config.max_data_points:
            idx = len(self.timestamps) - self.config.max_data_points
            self.timestamps = self.timestamps[idx:]
            self.itqan_scores = self.itqan_scores[idx:]
            self.mizan_scores = self.mizan_scores[idx:]
            self.tafakkur_scores = self.tafakkur_scores[idx:]
    
    def update_plots(self) -> None:
        """Update all plot visualizations."""
        self.update_data()
        
        # Clear all axes
        for ax in self.axes.values():
            ax.clear()
        
        x_indices = np.arange(len(self.timestamps))
        
        # Main combined plot
        ax = self.axes['main']
        ax.plot(x_indices, self.itqan_scores, 'o-', label='ITQAN', linewidth=2, markersize=6)
        ax.plot(x_indices, self.mizan_scores, 's-', label='MIZAN', linewidth=2, markersize=6)
        ax.plot(x_indices, self.tafakkur_scores, '^-', label='TAFAKKUR', linewidth=2, markersize=6)
        ax.set_title('Combined Metrics Overview', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left')
        ax.grid(self.config.show_grid, alpha=0.3)
        
        # Individual metric plots
        if self.itqan_scores:
            ax = self.axes['itqan']
            ax.plot(x_indices, self.itqan_scores, 'o-', color='#1f77b4', linewidth=2)
            ax.fill_between(x_indices, self.itqan_scores, alpha=0.3, color='#1f77b4')
            ax.set_title('ITQAN: Integrity & Quality', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(self.config.show_grid, alpha=0.3)
        
        if self.mizan_scores:
            ax = self.axes['mizan']
            ax.plot(x_indices, self.mizan_scores, 's-', color='#ff7f0e', linewidth=2)
            ax.fill_between(x_indices, self.mizan_scores, alpha=0.3, color='#ff7f0e')
            ax.set_title('MIZAN: Balance & Harmony', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(self.config.show_grid, alpha=0.3)
        
        if self.tafakkur_scores:
            ax = self.axes['tafakkur']
            ax.plot(x_indices, self.tafakkur_scores, '^-', color='#2ca02c', linewidth=2)
            ax.fill_between(x_indices, self.tafakkur_scores, alpha=0.3, color='#2ca02c')
            ax.set_title('TAFAKKUR: Reflection & Analysis', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(self.config.show_grid, alpha=0.3)
        
        # Statistics
        ax = self.axes['stats']
        ax.axis('off')
        
        stats_text = "MQIASE™ Metrics Summary\n" + "="*35 + "\n"
        
        if self.itqan_scores:
            stats_text += f"ITQAN: {self.itqan_scores[-1]:.4f}\n"
            stats_text += f"  Avg: {np.mean(self.itqan_scores):.4f} | "
            stats_text += f"Std: {np.std(self.itqan_scores):.4f}\n"
        
        if self.mizan_scores:
            stats_text += f"\nMIZAN: {self.mizan_scores[-1]:.4f}\n"
            stats_text += f"  Avg: {np.mean(self.mizan_scores):.4f} | "
            stats_text += f"Std: {np.std(self.mizan_scores):.4f}\n"
        
        if self.tafakkur_scores:
            stats_text += f"\nTAFAKKUR: {self.tafakkur_scores[-1]:.4f}\n"
            stats_text += f"  Avg: {np.mean(self.tafakkur_scores):.4f} | "
            stats_text += f"Std: {np.std(self.tafakkur_scores):.4f}\n"
        
        # Overall composite score
        if self.itqan_scores and self.mizan_scores and self.tafakkur_scores:
            composite = (self.itqan_scores[-1] + self.mizan_scores[-1] + self.tafakkur_scores[-1]) / 3.0
            stats_text += f"\nComposite Score: {composite:.4f}"
        
        stats_text += f"\n\nContact: ba.aj@hotmail.com"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def show(self) -> None:
        """Display the static visualization."""
        if self.fig is None:
            self.setup_figure()
        self.update_plots()
        plt.tight_layout()
        plt.show()
    
    def animate(self) -> None:
        """Show animated live updates (requires data to be added dynamically)."""
        if self.fig is None:
            self.setup_figure()
        
        self.animation_obj = animation.FuncAnimation(
            self.fig, lambda _: self.update_plots(),
            interval=self.config.update_interval,
            blit=False
        )
        
        plt.tight_layout()
        plt.show()


class MQIAASEDemo:
    """
    Complete MQIASE™ xAI-Ready Live Demo Controller.
    
    This class orchestrates the entire MQIASE system, managing metrics computation
    and visualization in a unified interface.
    """
    
    def __init__(self, name: str = "MQIASE™ xAI-Ready Live Demo v3.0"):
        """Initialize the MQIASE demo system."""
        self.name = name
        self.itqan = ITQANMetric()
        self.mizan = MIZANMetric()
        self.tafakkur = TAFAKKURMetric()
        self.visualizer = MQIAASELiveVisualizer(self.itqan, self.mizan, self.tafakkur)
        
        logger.info(f"Initialized {self.name}")
    
    def run_comprehensive_analysis(self,
                                   predictions: np.ndarray,
                                   ground_truth: np.ndarray,
                                   feature_importance: Optional[Union[np.ndarray, Dict[str, float]]] = None,
                                   fidelity: FidelityLevel = FidelityLevel.EXPERT) -> Dict[str, MetricResult]:
        """
        Run a comprehensive MQIASE analysis on the provided data.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            feature_importance: Optional feature importance scores
            fidelity: Fidelity level for all metrics
            
        Returns:
            Dictionary with all metric results
        """
        logger.info(f"Starting comprehensive analysis at fidelity level: {fidelity.name}")
        
        # Generate feature importance if not provided
        if feature_importance is None:
            feature_importance = np.random.rand(min(10, len(predictions)))
        
        # Compute all metrics
        itqan_result = self.itqan.compute(predictions, ground_truth, fidelity)
        mizan_result = self.mizan.compute(predictions, fidelity=fidelity)
        tafakkur_result = self.tafakkur.compute(feature_importance, fidelity=fidelity)
        
        results = {
            'itqan': itqan_result,
            'mizan': mizan_result,
            'tafakkur': tafakkur_result
        }
        
        # Compute composite score
        composite = (itqan_result.score + mizan_result.score + tafakkur_result.score) / 3.0
        logger.info(f"Composite MQIASE Score: {composite:.4f}")
        
        return results
    
    def visualize(self, show_animation: bool = False) -> None:
        """
        Visualize all metric results.
        
        Args:
            show_animation: If True, show animated visualization
        """
        if show_animation:
            self.visualizer.animate()
        else:
            self.visualizer.show()
    
    def get_report(self) -> str:
        """Generate a comprehensive text report of all metrics."""
        report = f"\n{'='*60}\n"
        report += f"{self.name}\n"
        report += f"Generated: {datetime.utcnow().isoformat()}\n"
        report += f"Contact: ba.aj@hotmail.com\n"
        report += f"{'='*60}\n\n"
        
        # ITQAN Report
        report += "ITQAN METRIC ANALYSIS\n"
        report += "-" * 40 + "\n"
        if self.itqan.history:
            latest = self.itqan.history[-1]
            report += f"Latest Score: {latest.score:.4f}\n"
            report += f"Fidelity Level: {latest.fidelity_level.name}\n"
            report += f"Confidence: {latest.confidence:.2%}\n"
            report += f"Components:\n"
            for comp_name, comp_value in latest.components.items():
                report += f"  - {comp_name}: {comp_value:.4f}\n"
            report += f"Average Score: {np.mean([r.score for r in self.itqan.history]):.4f}\n"
        report += "\n"
        
        # MIZAN Report
        report += "MIZAN METRIC ANALYSIS\n"
        report += "-" * 40 + "\n"
        if self.mizan.history:
            latest = self.mizan.history[-1]
            report += f"Latest Score: {latest.score:.4f}\n"
            report += f"Fidelity Level: {latest.fidelity_level.name}\n"
            report += f"Confidence: {latest.confidence:.2%}\n"
            report += f"Components:\n"
            for comp_name, comp_value in latest.components.items():
                report += f"  - {comp_name}: {comp_value:.4f}\n"
            report += f"Average Score: {np.mean([r.score for r in self.mizan.history]):.4f}\n"
        report += "\n"
        
        # TAFAKKUR Report
        report += "TAFAKKUR METRIC ANALYSIS\n"
        report += "-" * 40 + "\n"
        if self.tafakkur.history:
            latest = self.tafakkur.history[-1]
            report += f"Latest Score: {latest.score:.4f}\n"
            report += f"Fidelity Level: {latest.fidelity_level.name}\n"
            report += f"Confidence: {latest.confidence:.2%}\n"
            report += f"Components:\n"
            for comp_name, comp_value in latest.components.items():
                report += f"  - {comp_name}: {comp_value:.4f}\n"
            report += f"Average Score: {np.mean([r.score for r in self.tafakkur.history]):.4f}\n"
        report += "\n"
        
        # Composite Analysis
        report += "COMPOSITE ANALYSIS\n"
        report += "-" * 40 + "\n"
        if self.itqan.history and self.mizan.history and self.tafakkur.history:
            latest_composite = ((self.itqan.history[-1].score + 
                               self.mizan.history[-1].score + 
                               self.tafakkur.history[-1].score) / 3.0)
            report += f"Latest Composite Score: {latest_composite:.4f}\n"
            
            avg_composite = ((np.mean([r.score for r in self.itqan.history]) +
                            np.mean([r.score for r in self.mizan.history]) +
                            np.mean([r.score for r in self.tafakkur.history])) / 3.0)
            report += f"Average Composite Score: {avg_composite:.4f}\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report


# Example usage and demonstration
def main():
    """Main demonstration function."""
    print("\n" + "="*70)
    print("MQIASE™ xAI-Ready Live Demo v3.0")
    print("Advanced eXplainable Artificial Intelligence Framework")
    print("="*70)
    print("Contact: ba.aj@hotmail.com\n")
    
    # Initialize the demo
    demo = MQIAASEDemo()
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 50
    
    # Scenario 1: High-quality predictions
    print("Running Scenario 1: High-Quality Model Predictions")
    print("-" * 70)
    
    ground_truth = np.random.randint(0, 2, n_samples)
    predictions = ground_truth.astype(float) + np.random.randn(n_samples) * 0.1
    predictions = np.clip(predictions, 0, 1)
    
    feature_importance = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    results = demo.run_comprehensive_analysis(
        predictions=predictions,
        ground_truth=ground_truth,
        feature_importance=feature_importance,
        fidelity=FidelityLevel.ELITE
    )
    
    print("\nScenario 1 Results:")
    print(f"  ITQAN Score: {results['itqan'].score:.4f}")
    print(f"  MIZAN Score: {results['mizan'].score:.4f}")
    print(f"  TAFAKKUR Score: {results['tafakkur'].score:.4f}")
    print(f"  Composite: {(results['itqan'].score + results['mizan'].score + results['tafakkur'].score)/3:.4f}")
    
    # Scenario 2: Another analysis round
    print("\n\nRunning Scenario 2: Additional Model Analysis")
    print("-" * 70)
    
    ground_truth2 = np.random.randint(0, 3, n_samples)
    predictions2 = ground_truth2.astype(float) + np.random.randn(n_samples) * 0.2
    predictions2 = np.clip(predictions2, 0, 3)
    
    feature_importance2 = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    
    results2 = demo.run_comprehensive_analysis(
        predictions=predictions2,
        ground_truth=ground_truth2,
        feature_importance=feature_importance2,
        fidelity=FidelityLevel.ADVANCED
    )
    
    print("\nScenario 2 Results:")
    print(f"  ITQAN Score: {results2['itqan'].score:.4f}")
    print(f"  MIZAN Score: {results2['mizan'].score:.4f}")
    print(f"  TAFAKKUR Score: {results2['tafakkur'].score:.4f}")
    print(f"  Composite: {(results2['itqan'].score + results2['mizan'].score + results2['tafakkur'].score)/3:.4f}")
    
    # Generate comprehensive report
    print(demo.get_report())
    
    # Optional: Visualize results
    print("Generating visualizations...")
    demo.visualize(show_animation=False)
    
    print("\nDemo completed successfully!")
    print("Thank you for using MQIASE™ xAI-Ready Live Demo v3.0")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
