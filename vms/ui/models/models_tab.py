"""
Models tab for Video Model Studio UI
Provides an overview of all models and their statuses
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional, Tuple

from vms.utils.base_tab import BaseTab
from vms.ui.models.tabs import DraftsTab, TrainingTab, TrainedTab
from vms.ui.models.services import ModelsService

logger = logging.getLogger(__name__)

class ModelsTab(BaseTab):
    """Models tab for tracking all models"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "models_tab"
        self.title = "🎞️ Models"
        
        # Initialize service
        self.models_service = ModelsService(app_state)
        
        # Initialize sub-tabs
        self.drafts_tab = DraftsTab(app_state)
        self.training_tab = TrainingTab(app_state)
        self.trained_tab = TrainedTab(app_state)
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Models tab UI components"""
        with gr.Tab(self.title, id=self.id) as tab:
            # Create sub-tabs
            with gr.Tabs() as models_tabs:
                # Store reference to tabs component
                self.models_tabs_component = models_tabs
                
                # Create each sub-tab
                self.drafts_tab.create(models_tabs)
                self.training_tab.create(models_tabs)
                self.trained_tab.create(models_tabs)
        
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Connect events for each sub-tab
        self.drafts_tab.connect_events()
        self.training_tab.connect_events()
        self.trained_tab.connect_events()