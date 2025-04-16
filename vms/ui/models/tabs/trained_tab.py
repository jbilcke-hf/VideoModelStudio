"""
Trained tab for Models view in Video Model Studio UI
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional, Tuple
from gradio_modal import Modal

from vms.utils.base_tab import BaseTab

logger = logging.getLogger(__name__)

class TrainedTab(BaseTab):
    """Tab for managing trained models"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "trained_tab"
        self.title = "Trained"
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Trained tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                gr.Markdown("## Completed Models")
            
            # List for displaying models
            with gr.Column() as models_container:
                self.components["models_container"] = models_container
                self.components["no_models_message"] = gr.Markdown(
                    "No trained models found yet. Train a model to see it here.",
                    visible=False
                )
                
                # Placeholder for model rows - will be filled dynamically
                self.components["model_rows"] = []
                
                # Initial load of models
                self.refresh_models()
                
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Add auto-refresh timer
        refresh_timer = gr.Timer(interval=5)  # Refresh every 5 seconds
        refresh_timer.tick(
            fn=self.refresh_models,
            inputs=[],
            outputs=[self.components["models_container"]]
        )
    
    def refresh_models(self) -> gr.Column:
        """Refresh the list of trained models"""
        # Get models from service
        trained_models = self.app.models_tab.models_service.get_trained_models()
        
        # Create a new Column to replace the existing one
        with gr.Column() as new_container:
            if not trained_models:
                gr.Markdown("No trained models found yet. Train a model to see it here.")
            else:
                gr.Markdown(f"Found {len(trained_models)} trained models:")
                
                # Create headers
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=20):
                        gr.Markdown("### Model ID")
                    with gr.Column(scale=2, min_width=20):
                        gr.Markdown("### Model Type")
                    with gr.Column(scale=1, min_width=20):
                        gr.Markdown("### Completed")
                    with gr.Column(scale=2, min_width=20):
                        gr.Markdown("### Actions")
                
                # Create a row for each model
                for model in trained_models:
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2, min_width=20):
                            gr.Markdown(model.id[:8] + "...")
                        with gr.Column(scale=2, min_width=20):
                            gr.Markdown(model.model_display_name or "Unknown")
                        with gr.Column(scale=1, min_width=20):
                            gr.Markdown(model.updated_at.strftime("%Y-%m-%d"))
                            
                        with gr.Column(scale=2, min_width=20):
                            with gr.Row():
                                preview_btn = gr.Button("👁️ Preview", size="sm")
                                download_btn = gr.Button("💾 Download", size="sm")
                                publish_btn = gr.Button("🌐 Publish", size="sm")
                                delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                                
                                # Create a modal for this specific model deletion
                                with Modal(visible=False) as delete_modal:
                                    gr.Markdown("## ⚠️ Confirm Deletion")
                                    gr.Markdown(f"Are you sure you want to delete model {model.id[:8]}...?")
                                    gr.Markdown("This action cannot be undone!")
                                    
                                    with gr.Row():
                                        cancel_btn = gr.Button("🫢 No, cancel", variant="secondary")
                                        confirm_btn = gr.Button("🚨 Yes, delete", variant="primary")

                                # Connect event handlers for this specific model
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
                                
                                publish_btn.click(
                                    fn=lambda model_id=model.id: self.publish_model(model_id),
                                    inputs=[],
                                    outputs=[self.app.main_tabs]
                                )
                                
                                # Connect delete button to show modal
                                delete_btn.click(
                                    fn=lambda: Modal(visible=True),
                                    inputs=[],
                                    outputs=[delete_modal]
                                )
                                
                                # Connect cancel button to hide modal
                                cancel_btn.click(
                                    fn=lambda: Modal(visible=False),
                                    inputs=[],
                                    outputs=[delete_modal]
                                )
                                
                                # Connect confirm button to delete and hide modal
                                confirm_btn.click(
                                    fn=lambda model_id=model.id: self.delete_model(model_id),
                                    inputs=[],
                                    outputs=[new_container]
                                ).then(
                                    fn=lambda: Modal(visible=False),
                                    inputs=[],
                                    outputs=[delete_modal]
                                )
        
        return new_container
    
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
        
    def publish_model(self, model_id: str) -> gr.Tabs:
        """Publish model to Hugging Face Hub"""
        if self.app:
            # Switch to the selected model project
            self.app.switch_project(model_id)
            # Navigate to the main project tab
            return self.app.main_tabs.update(selected=0)
            # TODO: Navigate to publish tab
        
    def delete_model(self, model_id: str) -> gr.Column:
        """Delete a model and refresh the list"""
        if self.app and self.app.models_tab.models_service.delete_model(model_id):
            gr.Info(f"Model {model_id[:8]}... deleted successfully")
        else:
            gr.Warning(f"Failed to delete model {model_id[:8]}...")
            
        # Refresh the models list
        return self.refresh_models()