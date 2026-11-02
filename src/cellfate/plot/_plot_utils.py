from matplotlib.colors import to_rgb, to_hex, LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

def darken_color(color, amount=0.2):
    rgb = to_rgb(color)
    darker_rgb = [max(0, c - amount) for c in rgb]
    return to_hex(darker_rgb)

def create_handles(unique_values, color_dict, marker='o'):
    return [Line2D([0], [0], marker=marker, color='w', label=val, markersize=8, markerfacecolor=color_dict[val], markeredgecolor='black') for val in unique_values]

def make_cmap(values, colormap=None, custom_colors=None):
    """
    Generate a color mapping for given values using either a predefined colormap
    or a set of custom colors with interpolation.

    Parameters:
    ----------
    values : list
        List of unique values to assign colors to.
    colormap : str or matplotlib.colors.Colormap, optional
        Name of a matplotlib colormap or a Colormap object.
    custom_colors : list, optional
        List of custom colors (as hex or named) to interpolate.
    n_colors : int, optional
        Number of colors to generate if using custom interpolation.

    Returns:
    -------
    color_dict : dict
        Dictionary mapping values to corresponding colors.
    """
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    if len(set(values)) != len(values):
        raise ValueError('There are duplicate entries in the `values` list. Ensure all values are unique.')
    num_values = len(values)
    if colormap:
        cmap = plt.get_cmap(colormap)
        if cmap.N <= 30:
            cmap = LinearSegmentedColormap.from_list('custom_cmap', [cmap(i) for i in range(10)], N=num_values)
        colors = [to_hex(cmap(i / (num_values - 1))) for i in range(num_values)]
    elif custom_colors:
        if len(custom_colors) < 2:
            raise ValueError('At least two custom colors are required for interpolation.')
        rgb_colors = np.array([to_rgb(c) for c in custom_colors])
        x = np.linspace(0, 1, len(rgb_colors))
        interpolator = interp1d(x, rgb_colors, axis=0)
        new_x = np.linspace(0, 1, num_values)
        interpolated_colors = interpolator(new_x)
        colors = [to_hex(c) for c in interpolated_colors]
    else:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = default_colors[:num_values]
    color_dict = {val: col for (val, col) in zip(values, colors)}
    return color_dict
