"""Tensorflow implementation of the image segmenter"""
from typing import Optional
from hashlib import md5
import logging

from skimage import color
import tensorflow as tf
import numpy as np


from rtdefects.segmentation import download_model, model_dir, SemanticSegmenter

logger = logging.getLogger(__name__)

# Global variables for holding the model in memory
_model: Optional[tf.keras.models.Model] = None
_model_path = model_dir / 'segment-model.h5'


class TFSegmenter(SemanticSegmenter):
    """Tensorflow implementation of segmentation"""

    def transform_standard_image(self, image_data: np.ndarray) -> np.ndarray:
        image = color.gray2rgb(image_data)  # Convert to RGB
        image = np.array(image, dtype=np.float32) / 255  # Convert to float32
        return np.expand_dims(image, axis=0)

    def generate_mask(self, image_data: np.ndarray) -> np.ndarray:
        global _model  # Stores a loaded model between runs

        # Check the shape
        assert image_data.ndim == 4, "Expects a stack of images"
        assert image_data.shape[-1] == 3, "Expects 3 output channels"
        assert image_data.dtype == np.float32, "Expects np.float32"
        assert 0 <= np.min(image_data) and np.max(image_data) <= 1, "Image values should be in [0, 1]"

        # Download the model if needed
        if not _model_path.is_file():
            download_model('segment-model.h5')

        # If needed, load the model
        if _model is None:
            # Make sure the model exists
            if not _model_path.is_file():
                raise ValueError(f'Cannot find the model. No such file: {_model_path}')

            # Get the model hash to help with reproducibility
            with open(_model_path, 'rb') as fp:
                hsh = md5()
                while len(line := fp.read(4096 * 1024)) > 0:
                    hsh.update(line)
            logger.info(f'Loading the model from {_model_path}. MD5 Hash: {hsh.hexdigest()}')

            # Load it with Keras
            _model = tf.keras.models.load_model(_model_path)
            logger.info('Model loaded.')

        # Perform the segmentation
        return np.squeeze(_model.predict(image_data))
