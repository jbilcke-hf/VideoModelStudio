"""
Training tab for Models view in Video Model Studio UI
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional, Tuple

from vms.utils.base_tab import BaseTab

logger = logging.getLogger(__name__)

class TrainingTab(BaseTab):
    """Tab for managing models in training"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "training_tab"
        self.title = "Training"
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Training tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                gr.Markdown("## Models in Training")
            
            # List for displaying models
            with gr.Column() as models_container:
                self.components["models_container"] = models_container
                self.components["no_models_message"] = gr.Markdown(
                    "No models currently in training.",
                    visible=False
                )
                
                # Placeholder for model rows - will be filled dynamically
                self.components["model_rows"] = []
                
                # Initial load of models
                self.refresh_models()
                
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Add auto-refresh timer (no checkbox dependency)
        refresh_timer = gr.Timer(interval=5)  # Check training status every 5 seconds
        refresh_timer.tick(
            fn=self.refresh_models,  # Call directly without checking enabled flag
            inputs=[],
            outputs=[self.components["models_container"]]
        )
        
    def auto_refresh(self, enabled: bool) -> Optional[gr.Column]:
        """Auto-refresh if enabled"""
        if enabled:
            return self.refresh_models()
        return None
    
    def refresh_models(self) -> gr.Column:
        """Refresh the list of models in training"""
        # Get models from service
        training_models = self.app.models_tab.models_service.get_training_models()
        
        # Create a new Column to replace the existing one
        with gr.Column() as new_container:
            if not training_models:
                gr.Markdown("No models currently in training.")
            else:
                gr.Markdown(f"Found {len(training_models)} models in training:")
                
                # Create headers
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=20):
                        gr.Markdown("### Model ID")
                    with gr.Column(scale=1, min_width=20):
                        gr.Markdown("### Model Type")
                    with gr.Column(scale=2, min_width=20):
                        gr.Markdown("### Progress")
                    with gr.Column(scale=2, min_width=20):
                        gr.Markdown("### Actions")
                
                # Create a row for each model
                for model in training_models:
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=20):
                            gr.Markdown(model.id[:8] + "...")
                        with gr.Column(scale=1, min_width=20):
                            gr.Markdown(model.model_display_name or "Unknown")
                        
                        with gr.Column(scale=2, min_width=20):
                            progress_text = f"Step {model.current_step}/{model.total_steps} ({model.training_progress:.1f}%)"
                            gr.Markdown(progress_text)
                        
                        with gr.Column(scale=2, min_width=20):
                            with gr.Row():
                                stop_btn = gr.Button("⏹️ Stop", size="sm", variant="secondary")
                                preview_btn = gr.Button("👁️ Preview", size="sm")
                                download_btn = gr.Button("💾 Download", size="sm")
                                delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                                
                                # Connect event handlers for this specific model
                                stop_btn.click(
                                    fn=lambda model_id=model.id: self.stop_training(model_id),
                                    inputs=[],
                                    outputs=[new_container]
                                )
                                
                                preview_btn.click(
                                    fn=lambda model_id=model.id: self.preview_model(model_id),
                                    inputs=[],
                                    outputs=[self.app.main_tabs]
                                )
                                
                                download_btn.click(
                                    fn=lambda model_id=model.id: self.download_model(model_id),
                                    inputs=[],
                                    outputs=[]
                                )
                                
                                delete_btn.click(
                                    fn=lambda model_id=model.id: self.delete_model(model_id),
                                    inputs=[],
                                    outputs=[new_container]
                                )
        
        return new_container
    
    def stop_training(self, model_id: str) -> gr.Column:
        """Stop training for a model"""
        if self.app:
            # Save current project ID
            current_project = self.app.current_model_project_id
            
            # Switch to the model to stop
            self.app.switch_project(model_id)
            
            # Stop training
            result = self.app.training.stop_training()
            
            # Switch back to original project
            self.app.switch_project(current_project)
            
            # Show result message
            gr.Info(f"Training for model {model_id[:8]}... has been stopped.")
        
        # Refresh the list
        return self.refresh_models()
    
    def preview_model(self, model_id: str) -> gr.Tabs:
        """Open model preview"""
        if self.app:
            # Switch to project view with this model
            self.app.switch_project(model_id)
            # Set main tab to Project (index 0)
            return self.app.main_tabs.update(selected=0)
            # TODO: Navigate to preview tab
            
    def download_model(self, model_id: str) -> None:
        """Download model weights"""
        # TODO: Implement file download
        gr.Info(f"Download for model {model_id[:8]}... is not yet implemented")
        
    def delete_model(self, model_id: str) -> gr.Column:
        """Delete a model and refresh the list"""
        if self.app and self.app.models_tab.models_service.delete_model(model_id):
            gr.Info(f"Model {model_id[:8]}... deleted successfully")
        else:
            gr.Warning(f"Failed to delete model {model_id[:8]}...")
            
        # Refresh the models list
        return self.refresh_models()