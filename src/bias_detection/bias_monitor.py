"""
Bias Detection and Fairness Monitoring for Healthcare Face Recognition System
Implements fairness-aware machine learning and bias detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import sqlite3
from collections import defaultdict

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logging.warning("Fairlearn not available. Bias detection features will be limited.")

from config.settings import settings


class BiasMetric(Enum):
    """Bias detection metrics"""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    DISPARATE_IMPACT = "disparate_impact"
    STATISTICAL_PARITY = "statistical_parity"


class DemographicAttribute(Enum):
    """Demographic attributes for bias monitoring"""
    AGE = "age"
    GENDER = "gender"
    ETHNICITY = "ethnicity"
    RACE = "race"
    DISABILITY = "disability"


@dataclass
class BiasMeasurement:
    """Bias measurement result"""
    metric: BiasMetric
    attribute: DemographicAttribute
    value: float
    threshold: float
    is_biased: bool
    confidence: float
    timestamp: datetime
    details: Dict[str, Any]


@dataclass
class FairnessReport:
    """Comprehensive fairness report"""
    report_id: str
    timestamp: datetime
    overall_fairness_score: float
    bias_measurements: List[BiasMeasurement]
    recommendations: List[str]
    model_performance_by_group: Dict[str, Dict[str, float]]


class BiasMonitor:
    """
    Bias detection and fairness monitoring system
    Implements comprehensive bias detection across demographic groups
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conn = self._init_database()
        
        # Bias detection configuration
        self.bias_thresholds = {
            BiasMetric.DEMOGRAPHIC_PARITY: 0.1,
            BiasMetric.EQUALIZED_ODDS: 0.1,
            BiasMetric.EQUAL_OPPORTUNITY: 0.1,
            BiasMetric.DISPARATE_IMPACT: 0.8,
            BiasMetric.STATISTICAL_PARITY: 0.1
        }
        
        # Demographic group mappings
        self.demographic_groups = self._initialize_demographic_groups()
        
        # Performance tracking
        self.performance_history = defaultdict(list)
    
    def _init_database(self) -> sqlite3.Connection:
        """Initialize bias monitoring database"""
        try:
            # Extract database path from URL
            db_url = settings.database.database_url
            if db_url.startswith("sqlite:///"):
                db_path = db_url[10:]  # Remove "sqlite:///" prefix
            else:
                db_path = db_url
            conn = sqlite3.connect(db_path)
            
            # Create bias measurements table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bias_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT NOT NULL,
                    attribute TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    is_biased BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT
                )
            """)
            
            # Create fairness reports table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fairness_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    overall_fairness_score REAL NOT NULL,
                    bias_measurements TEXT,
                    recommendations TEXT,
                    model_performance TEXT
                )
            """)
            
            # Create demographic performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS demographic_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    demographic_group TEXT NOT NULL,
                    attribute TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    false_positive_rate REAL NOT NULL,
                    false_negative_rate REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            self.logger.info("Bias monitoring database initialized successfully")
            return conn
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bias monitoring database: {e}")
            raise
    
    def _initialize_demographic_groups(self) -> Dict[str, List[str]]:
        """Initialize demographic group mappings"""
        return {
            "age": ["0-18", "19-35", "36-50", "51-65", "65+"],
            "gender": ["male", "female", "non_binary", "other"],
            "ethnicity": ["hispanic", "non_hispanic"],
            "race": ["white", "black", "asian", "native_american", "pacific_islander", "other"],
            "disability": ["disabled", "non_disabled"]
        }
    
    def measure_bias(self, 
                    predictions: np.ndarray,
                    true_labels: np.ndarray,
                    demographic_data: Dict[str, np.ndarray],
                    model_name: str = "default") -> List[BiasMeasurement]:
        """
        Measure bias across different demographic groups
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            demographic_data: Demographic attributes for each sample
            model_name: Name of the model being evaluated
            
        Returns:
            List[BiasMeasurement]: Bias measurements
        """
        try:
            measurements = []
            
            # Convert to pandas DataFrame for easier manipulation
            df = pd.DataFrame({
                'predictions': predictions,
                'true_labels': true_labels
            })
            
            # Add demographic data
            for attr, values in demographic_data.items():
                df[attr] = values
            
            # Measure bias for each demographic attribute
            for attribute in DemographicAttribute:
                if attribute.value in df.columns:
                    attr_measurements = self._measure_attribute_bias(df, attribute, model_name)
                    measurements.extend(attr_measurements)
            
            # Store measurements in database
            self._store_bias_measurements(measurements)
            
            return measurements
            
        except Exception as e:
            self.logger.error(f"Failed to measure bias: {e}")
            return []
    
    def _measure_attribute_bias(self, df: pd.DataFrame, attribute: DemographicAttribute, 
                              model_name: str) -> List[BiasMeasurement]:
        """Measure bias for a specific demographic attribute"""
        try:
            measurements = []
            
            # Get unique groups for this attribute
            groups = df[attribute.value].unique()
            
            if len(groups) < 2:
                return measurements
            
            # Calculate bias metrics
            for metric in BiasMetric:
                try:
                    bias_value = self._calculate_bias_metric(df, attribute.value, metric)
                    threshold = self.bias_thresholds[metric]
                    is_biased = self._is_biased(bias_value, metric, threshold)
                    
                    measurement = BiasMeasurement(
                        metric=metric,
                        attribute=attribute,
                        value=bias_value,
                        threshold=threshold,
                        is_biased=is_biased,
                        confidence=self._calculate_confidence(df, attribute.value),
                        timestamp=datetime.now(),
                        details={
                            "model_name": model_name,
                            "groups": groups.tolist(),
                            "sample_size": len(df)
                        }
                    )
                    
                    measurements.append(measurement)
                    
                except Exception as e:
                    self.logger.error(f"Failed to calculate {metric.value} for {attribute.value}: {e}")
            
            return measurements
            
        except Exception as e:
            self.logger.error(f"Failed to measure bias for attribute {attribute.value}: {e}")
            return []
    
    def _calculate_bias_metric(self, df: pd.DataFrame, attribute: str, metric: BiasMetric) -> float:
        """Calculate specific bias metric"""
        try:
            if metric == BiasMetric.DEMOGRAPHIC_PARITY:
                return self._calculate_demographic_parity(df, attribute)
            elif metric == BiasMetric.EQUALIZED_ODDS:
                return self._calculate_equalized_odds(df, attribute)
            elif metric == BiasMetric.EQUAL_OPPORTUNITY:
                return self._calculate_equal_opportunity(df, attribute)
            elif metric == BiasMetric.DISPARATE_IMPACT:
                return self._calculate_disparate_impact(df, attribute)
            elif metric == BiasMetric.STATISTICAL_PARITY:
                return self._calculate_statistical_parity(df, attribute)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Failed to calculate {metric.value}: {e}")
            return 0.0
    
    def _calculate_demographic_parity(self, df: pd.DataFrame, attribute: str) -> float:
        """Calculate demographic parity difference"""
        try:
            if not FAIRLEARN_AVAILABLE:
                return self._simple_demographic_parity(df, attribute)
            
            # Use fairlearn for accurate calculation
            groups = df[attribute].values
            predictions = df['predictions'].values
            
            return demographic_parity_difference(
                y_true=None,  # Not needed for demographic parity
                y_pred=predictions,
                sensitive_features=groups
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate demographic parity: {e}")
            return 0.0
    
    def _simple_demographic_parity(self, df: pd.DataFrame, attribute: str) -> float:
        """Simple demographic parity calculation without fairlearn"""
        try:
            # Calculate positive prediction rate for each group
            group_rates = df.groupby(attribute)['predictions'].mean()
            
            # Calculate difference between max and min rates
            max_rate = group_rates.max()
            min_rate = group_rates.min()
            
            return max_rate - min_rate
            
        except Exception as e:
            self.logger.error(f"Failed to calculate simple demographic parity: {e}")
            return 0.0
    
    def _calculate_equalized_odds(self, df: pd.DataFrame, attribute: str) -> float:
        """Calculate equalized odds difference"""
        try:
            if not FAIRLEARN_AVAILABLE:
                return self._simple_equalized_odds(df, attribute)
            
            # Use fairlearn for accurate calculation
            groups = df[attribute].values
            predictions = df['predictions'].values
            true_labels = df['true_labels'].values
            
            return equalized_odds_difference(
                y_true=true_labels,
                y_pred=predictions,
                sensitive_features=groups
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate equalized odds: {e}")
            return 0.0
    
    def _simple_equalized_odds(self, df: pd.DataFrame, attribute: str) -> float:
        """Simple equalized odds calculation without fairlearn"""
        try:
            # Calculate TPR and FPR for each group
            group_metrics = {}
            
            for group in df[attribute].unique():
                group_df = df[df[attribute] == group]
                
                # True Positive Rate
                tp = ((group_df['predictions'] == 1) & (group_df['true_labels'] == 1)).sum()
                fn = ((group_df['predictions'] == 0) & (group_df['true_labels'] == 1)).sum()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # False Positive Rate
                fp = ((group_df['predictions'] == 1) & (group_df['true_labels'] == 0)).sum()
                tn = ((group_df['predictions'] == 0) & (group_df['true_labels'] == 0)).sum()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
            
            # Calculate maximum difference in TPR and FPR
            tprs = [metrics['tpr'] for metrics in group_metrics.values()]
            fprs = [metrics['fpr'] for metrics in group_metrics.values()]
            
            tpr_diff = max(tprs) - min(tprs)
            fpr_diff = max(fprs) - min(fprs)
            
            return max(tpr_diff, fpr_diff)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate simple equalized odds: {e}")
            return 0.0
    
    def _calculate_equal_opportunity(self, df: pd.DataFrame, attribute: str) -> float:
        """Calculate equal opportunity difference"""
        try:
            # Calculate TPR for each group
            group_tprs = {}
            
            for group in df[attribute].unique():
                group_df = df[df[attribute] == group]
                
                tp = ((group_df['predictions'] == 1) & (group_df['true_labels'] == 1)).sum()
                fn = ((group_df['predictions'] == 0) & (group_df['true_labels'] == 1)).sum()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                group_tprs[group] = tpr
            
            # Calculate difference between max and min TPR
            tprs = list(group_tprs.values())
            return max(tprs) - min(tprs)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate equal opportunity: {e}")
            return 0.0
    
    def _calculate_disparate_impact(self, df: pd.DataFrame, attribute: str) -> float:
        """Calculate disparate impact ratio"""
        try:
            # Calculate positive prediction rate for each group
            group_rates = df.groupby(attribute)['predictions'].mean()
            
            # Calculate ratio of minimum to maximum rate
            min_rate = group_rates.min()
            max_rate = group_rates.max()
            
            return min_rate / max_rate if max_rate > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate disparate impact: {e}")
            return 0.0
    
    def _calculate_statistical_parity(self, df: pd.DataFrame, attribute: str) -> float:
        """Calculate statistical parity difference"""
        try:
            # Same as demographic parity
            return self._calculate_demographic_parity(df, attribute)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate statistical parity: {e}")
            return 0.0
    
    def _is_biased(self, bias_value: float, metric: BiasMetric, threshold: float) -> bool:
        """Determine if bias value indicates bias"""
        try:
            if metric == BiasMetric.DISPARATE_IMPACT:
                # For disparate impact, lower values indicate more bias
                return bias_value < threshold
            else:
                # For other metrics, higher values indicate more bias
                return bias_value > threshold
                
        except Exception as e:
            self.logger.error(f"Failed to determine bias: {e}")
            return False
    
    def _calculate_confidence(self, df: pd.DataFrame, attribute: str) -> float:
        """Calculate confidence in bias measurement"""
        try:
            # Base confidence on sample size and group balance
            total_samples = len(df)
            group_counts = df[attribute].value_counts()
            
            # Calculate group balance
            min_group_size = group_counts.min()
            max_group_size = group_counts.max()
            balance_ratio = min_group_size / max_group_size if max_group_size > 0 else 0
            
            # Calculate confidence based on sample size and balance
            sample_confidence = min(1.0, total_samples / 1000.0)  # Full confidence at 1000 samples
            balance_confidence = balance_ratio
            
            return (sample_confidence + balance_confidence) / 2.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _store_bias_measurements(self, measurements: List[BiasMeasurement]):
        """Store bias measurements in database"""
        try:
            cursor = self.conn.cursor()
            
            for measurement in measurements:
                cursor.execute("""
                    INSERT INTO bias_measurements 
                    (metric, attribute, value, threshold, is_biased, confidence, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    measurement.metric.value,
                    measurement.attribute.value,
                    measurement.value,
                    measurement.threshold,
                    measurement.is_biased,
                    measurement.confidence,
                    json.dumps(measurement.details)
                ))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store bias measurements: {e}")
    
    def generate_fairness_report(self, model_name: str = "default") -> FairnessReport:
        """Generate comprehensive fairness report"""
        try:
            report_id = f"fairness_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get recent bias measurements
            recent_measurements = self._get_recent_bias_measurements(days=7)
            
            # Calculate overall fairness score
            overall_score = self._calculate_overall_fairness_score(recent_measurements)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(recent_measurements)
            
            # Get model performance by group
            performance_by_group = self._get_performance_by_group(model_name)
            
            report = FairnessReport(
                report_id=report_id,
                timestamp=datetime.now(),
                overall_fairness_score=overall_score,
                bias_measurements=recent_measurements,
                recommendations=recommendations,
                model_performance_by_group=performance_by_group
            )
            
            # Store report
            self._store_fairness_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate fairness report: {e}")
            return None
    
    def _get_recent_bias_measurements(self, days: int = 7) -> List[BiasMeasurement]:
        """Get recent bias measurements"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT metric, attribute, value, threshold, is_biased, confidence, timestamp, details
                FROM bias_measurements
                WHERE timestamp > datetime('now', '-{} days')
                ORDER BY timestamp DESC
            """.format(days))
            
            measurements = []
            for row in cursor.fetchall():
                measurement = BiasMeasurement(
                    metric=BiasMetric(row[0]),
                    attribute=DemographicAttribute(row[1]),
                    value=row[2],
                    threshold=row[3],
                    is_biased=bool(row[4]),
                    confidence=row[5],
                    timestamp=datetime.fromisoformat(row[6]),
                    details=json.loads(row[7]) if row[7] else {}
                )
                measurements.append(measurement)
            
            return measurements
            
        except Exception as e:
            self.logger.error(f"Failed to get recent bias measurements: {e}")
            return []
    
    def _calculate_overall_fairness_score(self, measurements: List[BiasMeasurement]) -> float:
        """Calculate overall fairness score"""
        try:
            if not measurements:
                return 0.5  # Neutral score if no measurements
            
            # Weight measurements by confidence
            total_weight = 0.0
            weighted_score = 0.0
            
            for measurement in measurements:
                weight = measurement.confidence
                score = 1.0 - (1.0 if measurement.is_biased else 0.0)
                
                weighted_score += score * weight
                total_weight += weight
            
            return weighted_score / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Failed to calculate overall fairness score: {e}")
            return 0.5
    
    def _generate_recommendations(self, measurements: List[BiasMeasurement]) -> List[str]:
        """Generate bias mitigation recommendations"""
        try:
            recommendations = []
            
            # Group measurements by attribute
            by_attribute = defaultdict(list)
            for measurement in measurements:
                by_attribute[measurement.attribute].append(measurement)
            
            # Generate recommendations for each attribute
            for attribute, attr_measurements in by_attribute.items():
                biased_count = sum(1 for m in attr_measurements if m.is_biased)
                
                if biased_count > 0:
                    recommendations.append(
                        f"Address bias in {attribute.value} attribute. "
                        f"{biased_count} out of {len(attr_measurements)} metrics show bias."
                    )
            
            # General recommendations
            if not recommendations:
                recommendations.append("No significant bias detected. Continue monitoring.")
            else:
                recommendations.extend([
                    "Consider retraining the model with more balanced data.",
                    "Implement bias mitigation techniques such as adversarial debiasing.",
                    "Regularly audit model performance across demographic groups."
                ])
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error."]
    
    def _get_performance_by_group(self, model_name: str) -> Dict[str, Dict[str, float]]:
        """Get model performance by demographic group"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT demographic_group, attribute, accuracy, precision, recall, f1_score,
                       false_positive_rate, false_negative_rate
                FROM demographic_performance
                WHERE timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
            """)
            
            performance = defaultdict(dict)
            for row in cursor.fetchall():
                group = row[0]
                attribute = row[1]
                metrics = {
                    'accuracy': row[2],
                    'precision': row[3],
                    'recall': row[4],
                    'f1_score': row[5],
                    'false_positive_rate': row[6],
                    'false_negative_rate': row[7]
                }
                performance[f"{attribute}_{group}"] = metrics
            
            return dict(performance)
            
        except Exception as e:
            self.logger.error(f"Failed to get performance by group: {e}")
            return {}
    
    def _store_fairness_report(self, report: FairnessReport):
        """Store fairness report in database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO fairness_reports 
                (report_id, overall_fairness_score, bias_measurements, recommendations, model_performance)
                VALUES (?, ?, ?, ?, ?)
            """, (
                report.report_id,
                report.overall_fairness_score,
                json.dumps([asdict(m) for m in report.bias_measurements]),
                json.dumps(report.recommendations),
                json.dumps(report.model_performance_by_group)
            ))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store fairness report: {e}")
    
    def track_demographic_performance(self, 
                                    demographic_group: str,
                                    attribute: str,
                                    performance_metrics: Dict[str, float]):
        """Track model performance by demographic group"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO demographic_performance 
                (demographic_group, attribute, accuracy, precision, recall, f1_score,
                 false_positive_rate, false_negative_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                demographic_group,
                attribute,
                performance_metrics.get('accuracy', 0.0),
                performance_metrics.get('precision', 0.0),
                performance_metrics.get('recall', 0.0),
                performance_metrics.get('f1_score', 0.0),
                performance_metrics.get('false_positive_rate', 0.0),
                performance_metrics.get('false_negative_rate', 0.0)
            ))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to track demographic performance: {e}")
    
    def get_bias_statistics(self) -> Dict:
        """Get bias detection statistics"""
        try:
            cursor = self.conn.cursor()
            
            # Get total measurements
            cursor.execute("SELECT COUNT(*) FROM bias_measurements")
            total_measurements = cursor.fetchone()[0]
            
            # Get biased measurements
            cursor.execute("SELECT COUNT(*) FROM bias_measurements WHERE is_biased = 1")
            biased_measurements = cursor.fetchone()[0]
            
            # Get measurements by metric
            cursor.execute("""
                SELECT metric, COUNT(*), AVG(value), AVG(confidence)
                FROM bias_measurements
                GROUP BY metric
            """)
            metric_stats = dict(cursor.fetchall())
            
            # Get recent reports
            cursor.execute("""
                SELECT COUNT(*), AVG(overall_fairness_score)
                FROM fairness_reports
                WHERE timestamp > datetime('now', '-30 days')
            """)
            recent_reports = cursor.fetchone()
            
            return {
                "total_measurements": total_measurements,
                "biased_measurements": biased_measurements,
                "bias_rate": biased_measurements / total_measurements if total_measurements > 0 else 0,
                "metric_statistics": metric_stats,
                "recent_reports_count": recent_reports[0],
                "average_fairness_score": recent_reports[1],
                "supported_metrics": [metric.value for metric in BiasMetric],
                "supported_attributes": [attr.value for attr in DemographicAttribute]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get bias statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
