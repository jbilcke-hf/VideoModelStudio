"""
Base class for UI tabs
"""

import gradio as gr
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseTab:
    """Base class for UI tabs with common functionality"""
    
    def __init__(self, app_state):
        """Initialize the tab with app state reference
        
        Args:
            app_state: Reference to main VideoTrainerUI instance
        """
        self.app = app_state
        self.components = {}
        
    def create(self, parent=None) -> gr.TabItem:
        """Create the tab UI components
        
        Args:
            parent: Optional parent container
            
        Returns:
            The created tab component
        """
        raise NotImplementedError("Subclasses must implement create()")
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        raise NotImplementedError("Subclasses must implement connect_events()")
    
    def refresh(self) -> Dict[str, Any]:
        """Refresh UI components with current data
        
        Returns:
            Dictionary with updated values for components
        """
        return {}