"""Comprehensive logging system for OpenRubricRL."""

import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from .storage import StorageBackend, get_default_storage
from ..core.reward_processor import ProcessedReward


class OpenRubricLogger:
    """Main logger for OpenRubricRL scoring and training."""
    
    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True
    ):
        self.storage = storage or get_default_storage()
        
        # Set up Python logging
        self.logger = logging.getLogger("openrubricrl")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(file_handler)
    
    def log_scoring_result(
        self,
        result: ProcessedReward,
        task_input: str,
        model_output: str,
        rubric_name: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a scoring result."""
        metadata = {
            "task_input": task_input,
            "model_output": model_output,
            "rubric_name": rubric_name,
            **(additional_metadata or {})
        }
        
        # Store in backend
        result_id = self.storage.store_result(result, metadata)
        
        # Log summary
        self.logger.info(
            f"Scored result {result_id}: reward={result.reward:.3f}, "
            f"llm_score={result.raw_llm_score:.3f}, rubric={rubric_name}"
        )
        
        # Log detailed explanation at debug level
        self.logger.debug(f"Result {result_id} explanation: {result.explanation}")
        
        return result_id
    
    def log_training_step(
        self,
        step: int,
        episode: int,
        rewards: List[float],
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Log training step metrics."""
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        self.logger.info(
            f"Step {step}, Episode {episode}: "
            f"avg_reward={avg_reward:.3f}, "
            f"min_reward={min(rewards) if rewards else 0:.3f}, "
            f"max_reward={max(rewards) if rewards else 0:.3f}"
        )
        
        if additional_metrics:
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in additional_metrics.items())
            self.logger.info(f"Additional metrics: {metrics_str}")
    
    def log_episode_summary(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """Log episode summary."""
        self.logger.info(
            f"Episode {episode} completed: "
            f"total_reward={total_reward:.3f}, "
            f"length={episode_length}"
        )
        
        if additional_info:
            self.logger.debug(f"Episode {episode} additional info: {json.dumps(additional_info)}")
    
    def log_error(self, error: Exception, context: Optional[str] = None):
        """Log an error with context."""
        error_msg = f"Error: {str(error)}"
        if context:
            error_msg = f"{context}: {error_msg}"
        
        self.logger.error(error_msg, exc_info=True)
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)
    
    def get_recent_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent scoring results."""
        return self.storage.query_results(limit=limit)
    
    def get_statistics(self, rubric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics."""
        return self.storage.get_statistics(rubric_name=rubric_name)
    
    def export_results(
        self,
        output_file: str,
        format: str = "json",
        filters: Optional[Dict[str, Any]] = None
    ):
        """Export results to file."""
        results = self.storage.query_results(filters=filters)
        
        output_path = Path(output_file)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format.lower() == "csv":
            try:
                import pandas as pd
                
                # Flatten results for CSV
                flattened = []
                for result in results:
                    flat_result = {
                        "id": result.get("id"),
                        "timestamp": result.get("timestamp"),
                        "rubric_name": result.get("rubric_name"),
                        "reward": result.get("reward"),
                        "raw_llm_score": result.get("raw_llm_score"),
                        "explanation": result.get("explanation")
                    }
                    
                    # Add hybrid components as separate columns
                    hybrid_components = result.get("hybrid_components", {})
                    if isinstance(hybrid_components, str):
                        hybrid_components = json.loads(hybrid_components)
                    
                    for metric, score in hybrid_components.items():
                        flat_result[f"hybrid_{metric}"] = score
                    
                    flattened.append(flat_result)
                
                df = pd.DataFrame(flattened)
                df.to_csv(output_path, index=False)
                
            except ImportError:
                raise ImportError("Install pandas for CSV export: pip install pandas")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported {len(results)} results to {output_path}")


class TrainingLogger:
    """Specialized logger for RL training sessions."""
    
    def __init__(
        self,
        session_name: str,
        base_logger: Optional[OpenRubricLogger] = None,
        metrics_file: Optional[str] = None
    ):
        self.session_name = session_name
        self.logger = base_logger or OpenRubricLogger()
        self.metrics_file = metrics_file
        
        self.session_start_time = datetime.now()
        self.episode_count = 0
        self.step_count = 0
        self.total_rewards = []
        self.episode_lengths = []
        
        # Log session start
        self.logger.log_info(f"Training session '{session_name}' started")
        
        # Initialize metrics file
        if metrics_file:
            with open(metrics_file, 'w') as f:
                f.write("timestamp,episode,step,reward,episode_length,avg_reward\n")
    
    def log_step(
        self,
        reward: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Log a training step."""
        self.step_count += 1
        
        # Log to main logger
        self.logger.log_debug(f"Step {self.step_count}: reward={reward:.3f}")
        
        # Write to metrics file
        if self.metrics_file:
            avg_reward = sum(self.total_rewards) / len(self.total_rewards) if self.total_rewards else 0.0
            
            with open(self.metrics_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()},{self.episode_count},"
                       f"{self.step_count},{reward},{len(self.episode_lengths)},"
                       f"{avg_reward}\n")
    
    def log_episode_end(
        self,
        total_reward: float,
        episode_length: int,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """Log the end of an episode."""
        self.episode_count += 1
        self.total_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        # Calculate running statistics
        avg_reward = sum(self.total_rewards) / len(self.total_rewards)
        avg_length = sum(self.episode_lengths) / len(self.episode_lengths)
        
        self.logger.log_episode_summary(
            episode=self.episode_count,
            total_reward=total_reward,
            episode_length=episode_length,
            additional_info=additional_info
        )
        
        # Log running averages every 10 episodes
        if self.episode_count % 10 == 0:
            self.logger.log_info(
                f"Running averages (last {len(self.total_rewards)} episodes): "
                f"avg_reward={avg_reward:.3f}, avg_length={avg_length:.1f}"
            )
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the training session."""
        session_duration = datetime.now() - self.session_start_time
        
        return {
            "session_name": self.session_name,
            "duration_seconds": session_duration.total_seconds(),
            "episodes_completed": self.episode_count,
            "total_steps": self.step_count,
            "avg_reward": sum(self.total_rewards) / len(self.total_rewards) if self.total_rewards else 0.0,
            "avg_episode_length": sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0,
            "best_episode_reward": max(self.total_rewards) if self.total_rewards else 0.0,
            "worst_episode_reward": min(self.total_rewards) if self.total_rewards else 0.0
        }
    
    def end_session(self):
        """End the training session and log summary."""
        summary = self.get_session_summary()
        
        self.logger.log_info(f"Training session '{self.session_name}' ended")
        self.logger.log_info(f"Session summary: {json.dumps(summary, indent=2)}")


# Global logger instance
_default_logger = None


def get_default_logger() -> OpenRubricLogger:
    """Get the default logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = OpenRubricLogger()
    return _default_logger


def set_default_logger(logger: OpenRubricLogger) -> None:
    """Set the default logger instance."""
    global _default_logger
    _default_logger = logger


# Convenience functions
def log_scoring_result(
    result: ProcessedReward,
    task_input: str,
    model_output: str,
    rubric_name: str,
    **kwargs
) -> str:
    """Log a scoring result using the default logger."""
    return get_default_logger().log_scoring_result(
        result, task_input, model_output, rubric_name, **kwargs
    )


def log_training_step(step: int, episode: int, rewards: List[float], **kwargs):
    """Log a training step using the default logger."""
    get_default_logger().log_training_step(step, episode, rewards, **kwargs)


def get_statistics(rubric_name: Optional[str] = None) -> Dict[str, Any]:
    """Get statistics using the default logger."""
    return get_default_logger().get_statistics(rubric_name)