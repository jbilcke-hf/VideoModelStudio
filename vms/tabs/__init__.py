"""
Tab components for Video Model Studio UI
"""

from .base_tab import BaseTab
from .import_tab import ImportTab
from .split_tab import SplitTab
from .caption_tab import CaptionTab
from .train_tab import TrainTab
from .monitor_tab import MonitorTab
from .preview_tab import PreviewTab
from .manage_tab import ManageTab

__all__ = [
    'BaseTab',
    'ImportTab',
    'SplitTab',
    'CaptionTab',
    'TrainTab',
    'MonitorTab',
    'PreviewTab',
    'ManageTab'
]