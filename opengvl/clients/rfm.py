"""RFM (Reward Foundation Model) client for OpenGVL.

This client integrates trained RFM models into the OpenGVL evaluation framework.
RFM models predict progress values (0-1) for video frames, which are converted
to completion percentages (0-100) for OpenGVL compatibility.
"""

from typing import cast, List

import numpy as np
from loguru import logger
from PIL import Image

from opengvl.clients.base import BaseModelClient
from opengvl.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from rfm.evals.baselines.rfm_model import RFMModel


class RFMClient(BaseModelClient):
    """RFM client that uses trained RFM models for progress prediction.
    
    This client loads an RFM model from a checkpoint and uses it to predict
    task completion percentages for video frames. The model expects frames
    as numpy arrays or PIL Images and returns progress values (0-1), which
    are converted to percentages (0-100) for OpenGVL compatibility.
    """

    def __init__(self, *, rpm: float = 0.0, model_path: str):
        """Initialize the RFM client.
        
        Args:
            rpm: Requests per minute rate limit (0.0 for no limit).
            model_path: Path to RFM model checkpoint (HuggingFace repo ID or local path).
        """
        super().__init__(rpm=rpm)
        
        if not model_path:
            raise ValueError("model_path is required for RFMClient")
        
        self.model_path = model_path
        logger.info(f"Loading RFM model from: {model_path}")
        
        # Load the RFM model
        self.rfm_model = RFMModel(checkpoint_path=model_path)
        logger.info(f"RFM model loaded successfully")
        
        # Extract model name for identification
        self.model_name = model_path.replace("/", "_").replace("\\", "_")

    def _generate_response_impl(
        self,
        prompt: str,
        eval_episode,
        context_episodes: list,
        temperature: float = 0.0,
        *,
        prompt_phrases: dict[str, str],
    ) -> str:
        """Generate progress predictions using RFM model.
        
        This method extracts frames from the evaluation episode,
        calls the RFM model to get progress predictions, and formats
        the output as text with completion percentages.
        
        Args:
            prompt: Base prompt text (not used by RFM).
            eval_episode: Episode object containing frames and instruction.
            context_episodes: Context episodes (not used by RFM).
            temperature: Temperature parameter (not used by RFM).
            prompt_phrases: Prompt phrase dictionary (not used by RFM).
            
        Returns:
            Formatted string with progress predictions like:
            "Frame 1: Task Completion: 50%\nFrame 2: Task Completion: 100%"
        """
        # Extract task instruction from episode
        task_instruction = eval_episode.instruction if hasattr(eval_episode, 'instruction') else ""
        
        # Extract frames from the evaluation episode
        # Use shuffled_frames as that's what OpenGVL expects predictions for
        frames = []
        for frame in eval_episode.shuffled_frames:
            img = frame
            # Convert to numpy array if needed
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            elif isinstance(img, np.ndarray):
                img_array = img
            else:
                # Convert tensor-like objects
                img_array = np.array(img)
            
            # Ensure uint8 format and correct shape
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            # Ensure 3D array (H, W, C)
            if img_array.ndim == 2:
                # Grayscale, convert to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.ndim == 4:
                # Batch dimension, take first
                img_array = img_array[0]
            
            frames.append(img_array)
        
        if len(frames) == 0:
            logger.error("No frames found in evaluation episode")
            return "Frame 1: Task Completion: 0%"
        
        logger.info(f"RFM predicting progress for {len(frames)} frames, task: {task_instruction[:50]}...")
        
        import ipdb; ipdb.set_trace()
        # Call RFM model to get progress predictions
        progress_values = self.rfm_model.compute_progress(
            frames_array=frames,
            task_description=task_instruction
        )
        
        # Ensure we have the right number of predictions
        if len(progress_values) != len(frames):
            logger.warning(
                f"RFM returned {len(progress_values)} predictions for {len(frames)} frames. "
                f"Padding or truncating as needed."
            )
            if len(progress_values) < len(frames):
                # Pad with last value
                progress_values.extend([progress_values[-1]] * (len(frames) - len(progress_values)))
            else:
                # Truncate
                progress_values = progress_values[:len(frames)]
        
        # Convert progress values (0-1) to percentages (0-100) and format output
        # RFM returns progress as 0-1, we need to convert to 0-100 for OpenGVL
        # Output frame numbers should be sequential (1, 2, 3, ...) matching the order
        # of shuffled_frames, which is what we're predicting for
        output_lines = []
        for i, progress in enumerate(progress_values):
            # Clamp progress to [0, 1] range
            progress = max(0.0, min(1.0, float(progress)))
            percentage = int(round(progress * 100))
            # Use sequential frame numbers (1-indexed) matching the shuffled order
            frame_num = i + 1
            output_lines.append(f"Frame {frame_num}: Task Completion: {percentage}%")
        
        result = "\n".join(output_lines)
        logger.debug(f"RFM prediction result (first 200 chars): {result[:200]}...")
        return result

    def _generate_from_events(self, events: List[Event], temperature: float) -> str:
        """RFM doesn't use the event-based interface.
        
        This method is required by the abstract base class but is not used
        since RFMClient overrides _generate_response_impl directly.
        """
        # This should never be called since we override _generate_response_impl
        raise NotImplementedError(
            "RFMClient uses _generate_response_impl directly and does not use the event-based interface"
        )

