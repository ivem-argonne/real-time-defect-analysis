"""Functions to analyze segmented images"""
import logging
from warnings import warn
from typing import Iterator

from skimage import measure, morphology
from scipy.stats import siegelslopes
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def analyze_defects(labeled_mask: np.ndarray, edge_buffer: int = 8, defect_names: list[str] | None = None) -> dict:
    """Analyze the voids in a masked image

    Args:
        labeled_mask: Mask for each of the segmented defects. Dimensions: <defect> x <width> x <height>
        edge_buffer: Label voids as touching edge if they are within this many pixels of the side
        defect_names: Names of defects, if available
    Returns:
        Dictionary of the computed properties
    """

    # Special case: Make a blank image if there are no defects
    if labeled_mask.shape[0] == 0:
        labeled_mask = np.zeros((1, *labeled_mask.shape[-2:]), dtype=labeled_mask.dtype)

    # Basic statistics
    output = {
        'void_frac': (labeled_mask > 0).sum() / (labeled_mask.shape[0] * labeled_mask.shape[1]),
        'void_count': labeled_mask.shape[0]
    }

    # Create a 2D version of the image, where the pixel value is which defect that region "belongs to"
    if (labeled_mask > 0).sum(axis=0).max() > 1:
        warn('Some instances in this image overlap which each other. Talk to Logan about fixing it')
    object_inds = np.arange(labeled_mask.shape[0]) + 1
    with_index_3d = (labeled_mask > 0) * object_inds[:, None, None]
    with_index = with_index_3d.max(axis=0)

    # Compute region properties
    props = measure.regionprops(with_index, (with_index > 0))
    radii = [p['equivalent_diameter'] / 2 for p in props]
    output['type'] = labeled_mask.max(axis=(1, 2)).tolist()
    output['radii'] = radii
    output['radii_average'] = np.average(radii)
    output['positions'] = [p['centroid'][::-1] for p in props]  # From (y, x) to (x, y)

    # Use the name of the defect type, if known
    if defect_names is not None:
        output['type'] = [defect_names[t - 1] for t in output['type']]

    # Determine if it touches the side
    output['touches_side'] = [
        min(p['bbox']) <= edge_buffer
        or p['bbox'][2] >= labeled_mask.shape[1] - edge_buffer
        or p['bbox'][3] >= labeled_mask.shape[2] - edge_buffer
        for p in props
    ]

    return output


def label_instances_from_mask(mask: np.array, min_size: int = 50) -> np.ndarray:
    """Label distinct instances of defects within a larger void

    Args:
        mask: Boolean mask of isolated features
        min_size: Minimum size of defect (units: pixels)
    Returns:
        Image where distinct regions are labeled with different positive integers.
        Numpy array with be uint8.
    """

    # Clean up the mask
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, min_size)
    mask = morphology.binary_erosion(mask, morphology.square(1))

    # Assign labels to the distinct regions
    output = measure.label(mask)
    return np.array(output, dtype=np.uint16)


def convert_to_per_particle(per_frame: pd.DataFrame, position_col: str = 'positions', exclude_column: str | None = None) -> Iterator[pd.DataFrame]:
    """Convert the per-frame void information to the per-particle format expected by trackpy

    Args:
        per_frame: A DataFrame where each row is a different image and
            contains the defect locations in `positions` and sizes in `radii` columns.
        position_col: Name of the column holding positions of the particles
        exclude_column: If provided, exclude frames where the value of this column is true from output
    Yields:
        A dataframe where each row is a different defect
    """

    for rid, row in per_frame.iterrows():
        if exclude_column is not None and row[exclude_column]:
            continue
        particles = pd.DataFrame(row[position_col], columns=['x', 'y'])
        particles['local_id'] = np.arange(len(row[position_col]))
        particles['frame'] = rid
        particles['radius'] = row['radii']
        particles['touches_side'] = row['touches_side']
        yield particles


def compile_void_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Compile summary statistics about each void

    Args:
        tracks: Track information for each void over time

    Returns:
        Dataframe of the summary of voids
        - "start_frame": First frame in which the void appears
        - "end_frame": Last frame in which the void appears
        - "total_frames": Total number of frames in which the void appears
        - "positions": Positions of the void in each frame
        - "touches_side": Whether the void touches the side at this frame
        - "local_id": ID of the void in each frame (if available)
        - "disp_from_start": How far the void has moved from the first frame
        - "max_disp": Maximum distance the void moved
        - "drift_rate": Average displacement from center over time
        - "dist_traveled": Total path distance the void has traveled
        - "total_travel": How far the void traveled over its whole life
        - "movement_rate": How far the void moves per frame
        - "radii": Radius of the void in each frame
        - "max_radius": Maximum radius of the void
        - "min_radius": Minimum radius of the void
        - "growth_rate": Median rate of change of the radius
    """

    # Loop over all unique voids
    voids = []
    for t, track in tracks.groupby('particle'):
        # Get the frames where this void is visible
        visible_frames = track['frame']

        # Get all frames between start and stop
        frames_id = np.arange(track['frame'].min(), track['frame'].max() + 1)

        # Build an interpolator for position as a function of frame
        if len(track) == 1:
            positions = track[['x', 'y']].values
        else:
            x_inter = interp1d(track['frame'], track['x'])
            y_inter = interp1d(track['frame'], track['y'])

            # Compute the displacement over each step
            positions = [(x_inter(f), y_inter(f)) for f in frames_id]
            positions = np.array(positions)

        # Get the ID from each frame and whether it touches the side
        id_lookup = dict(zip(track['frame'], track['local_id']))
        local_id = [id_lookup.get(i, None) for i in frames_id]

        # Use interpolation to detect if any point used to interpolate
        #  the void position was on the side
        if len(track) == 1:
            touches_side = track['touches_side'].values
        else:
            ts_inter = interp1d(track['frame'], np.array(track['touches_side'], dtype=float), kind='linear')
            touches_side = ts_inter(frames_id) > 0  # It is only zero if neither point used in the left or right touches side (and equals 1)

        # Gather some basic information about the void
        void_info = {
            'start_frame': np.min(visible_frames),
            'end_frame': np.max(visible_frames),
            'total_frames': len(frames_id),
            'inferred_frames': len(frames_id) - len(track),
            'positions': positions,
            'touches_side': touches_side,
            'local_id': local_id
        }

        # If there is only one frame, we cannot do the following steps
        if positions.shape[0] > 1:
            # Compute the displacement from the start
            void_info['disp_from_start'] = np.linalg.norm(positions - positions[0, :], axis=1)
            void_info['max_disp'] = np.max(void_info['disp_from_start'])
            void_info['drift_rate'] = void_info['max_disp'] / void_info['total_frames']

            # Get the displacement for each step
            void_info['dist_traveled'] = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))))
            void_info['total_traveled'] = void_info['dist_traveled'][-1]
            void_info['movement_rate'] = void_info['total_traveled'] / void_info['total_frames']

        # More stats if we have radii
        if len(track) == 1:
            radii = track['radius'].values
        else:
            r_inter = interp1d(track['frame'], track['radius'])
            radii = r_inter(frames_id)

        # Store some summary information
        void_info['radii'] = radii
        void_info['max_radius'] = max(radii)
        void_info['min_radius'] = min(radii)
        if len(radii) > 3:
            void_info['growth_rate'] = siegelslopes(radii)[0]

        # Add it to list
        voids.append(void_info)
    return pd.DataFrame(voids)
