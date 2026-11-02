"""CellFate public API."""
from importlib import import_module

__version__ = '0.1.2'

_ATTRIBUTES = {
    'calculate_relative_dimension': ('rdim', 'calculate_relative_dimension'),
    'run_selected_sources': ('rdim', 'run_selected_sources'),
    'run_selected_sources_lazy_scipy_tiled': ('rdim', 'run_selected_sources_lazy_scipy_tiled'),
    'run_selected_sources_lazy_threaded': ('rdim', 'run_selected_sources_lazy_threaded'),
    'run_selected_sources_lazy_tiled': ('rdim', 'run_selected_sources_lazy_tiled'),
    'run_single_source': ('rdim', 'run_single_source'),
    'build_mic_candidates': ('mic', 'build_mic_candidates'),
    'calculate_MIC_atlas': ('mic', 'calculate_MIC_atlas'),
    'calculate_MIC_candidate': ('mic', 'calculate_MIC_candidate'),
    'calculate_MIC_filtered': ('mic', 'calculate_MIC_filtered'),
    'generate_mi_mat': ('mic', 'generate_mi_mat'),
    'generate_graph': ('construct_graph', 'generate_graph'),
    'build_knn_adjacency': ('construct_graph', 'build_knn_adjacency'),
    'build_knn_graph': ('construct_graph', 'build_knn_graph'),
    'build_neighbor_candidate_pairs': ('construct_graph', 'build_neighbor_candidate_pairs'),
    'calculate_exact_mic_graph': ('construct_graph', 'calculate_exact_mic_graph'),
    'run_exact_fate_pipeline': ('construct_graph', 'run_exact_fate_pipeline'),
    'FatePredictor': ('stats', 'FatePredictor'),
}
_MODULES = {'mic': 'mic', 'rdim': 'rdim', 'construct_graph': 'construct_graph',
            'stats': 'stats', 'plot': 'plot', 'pl': 'plot',
            'atlas': 'construct_graph', 'latent_graph': 'construct_graph'}
__all__ = [*_ATTRIBUTES, 'pl', 'backend_info']

def __getattr__(name):
    if name in _MODULES:
        value = import_module('.' + _MODULES[name], __name__)
    elif name in _ATTRIBUTES:
        module, attribute = _ATTRIBUTES[name]
        value = getattr(import_module('.' + module, __name__), attribute)
    else:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    globals()[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(__all__))

def backend_info():
    from . import _native, utils
    return dict(minepy='1.2.6+cellfate6', minepy_binary=_native.mine.__file__,
                rd_kernel=utils.register_lib._name, rd_backend='register', block_size=16)

