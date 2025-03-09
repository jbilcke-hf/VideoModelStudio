"""
Hugging Face Hub tab for Video Model Studio UI.
Handles browsing, searching, and importing datasets from the Hugging Face Hub.
"""

import gradio as gr
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..base_tab import BaseTab

logger = logging.getLogger(__name__)

class HubTab(BaseTab):
    """Hub tab for importing datasets from Hugging Face Hub"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "hub_tab"
        self.title = "Import from Hugging Face"
    
    def create(self, parent=None) -> gr.Tab:
        """Create the Hub tab UI components"""
        with gr.Tab(self.title, id=self.id) as tab:
            with gr.Column():
                with gr.Row():
                    gr.Markdown("## Import from Hub datasets")
                
                with gr.Row():
                    gr.Markdown("Search for datasets with videos or WebDataset archives:")
                
                with gr.Row():
                    self.components["dataset_search"] = gr.Textbox(
                        label="Search Hugging Face Datasets",
                        placeholder="Search for video datasets..."
                    )
                
                with gr.Row():
                    self.components["dataset_search_btn"] = gr.Button("Search Datasets", variant="primary")
                
                # Dataset browser results section
                with gr.Row(visible=False) as dataset_results_row:
                    self.components["dataset_results_row"] = dataset_results_row
                    
                    with gr.Column(scale=3):
                        self.components["dataset_results"] = gr.Dataframe(
                            headers=["id", "title", "downloads"],
                            interactive=False,
                            wrap=True,
                            row_count=10,
                            label="Dataset Results"
                        )
                    
                    with gr.Column(scale=3):
                        # Dataset info and state
                        self.components["dataset_info"] = gr.Markdown("Select a dataset to see details")
                        self.components["dataset_id"] = gr.State(value=None)
                        self.components["file_type"] = gr.State(value=None)
                        
                        # Files section that appears when a dataset is selected
                        with gr.Column(visible=False) as files_section:
                            self.components["files_section"] = files_section
                            
                            gr.Markdown("## Files:")
                            
                            # Video files row (appears if videos are present)
                            with gr.Row(visible=False) as video_files_row:
                                self.components["video_files_row"] = video_files_row
                                
                                with gr.Column(scale=4):
                                    self.components["video_count_text"] = gr.Markdown("Contains 0 video files")
                                
                                with gr.Column(scale=1):
                                    self.components["download_videos_btn"] = gr.Button("Download", variant="primary")
                            
                            # WebDataset files row (appears if tar files are present)
                            with gr.Row(visible=False) as webdataset_files_row:
                                self.components["webdataset_files_row"] = webdataset_files_row
                                
                                with gr.Column(scale=4):
                                    self.components["webdataset_count_text"] = gr.Markdown("Contains 0 WebDataset (.tar) files")
                                
                                with gr.Column(scale=1):
                                    self.components["download_webdataset_btn"] = gr.Button("Download", variant="primary")
                        
                        # Status and loading indicators
                        self.components["dataset_loading"] = gr.Markdown(visible=False)
            
            return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Dataset search event
        self.components["dataset_search_btn"].click(
            fn=self.search_datasets,
            inputs=[self.components["dataset_search"]],
            outputs=[
                self.components["dataset_results"],
                self.components["dataset_results_row"]
            ]
        )
        
        # Dataset selection event - FIX HERE
        self.components["dataset_results"].select(
            fn=self.display_dataset_info,
            outputs=[
                self.components["dataset_info"],
                self.components["dataset_id"],
                self.components["files_section"],
                self.components["video_files_row"],
                self.components["video_count_text"],
                self.components["webdataset_files_row"],
                self.components["webdataset_count_text"]
            ]
        )
        
        # Download videos button
        self.components["download_videos_btn"].click(
            fn=self.set_file_type_and_return,
            outputs=[self.components["file_type"]]
        ).then(
            fn=self.download_file_group,
            inputs=[
                self.components["dataset_id"],
                self.components["enable_automatic_video_split"],
                self.components["file_type"]
            ],
            outputs=[
                self.components["dataset_loading"],
                self.components["import_status"]
            ]
        ).success(
            fn=self.app.tabs["import_tab"].on_import_success,
            inputs=[
                self.components["enable_automatic_video_split"],
                self.components["enable_automatic_content_captioning"],
                self.app.tabs["caption_tab"].components["custom_prompt_prefix"]
            ],
            outputs=[
                self.app.tabs_component,
                self.app.tabs["split_tab"].components["video_list"],
                self.app.tabs["split_tab"].components["detect_status"]
            ]
        )
        
        # Download WebDataset button
        self.components["download_webdataset_btn"].click(
            fn=self.set_file_type_and_return_webdataset,
            outputs=[self.components["file_type"]]
        ).then(
            fn=self.download_file_group,
            inputs=[
                self.components["dataset_id"],
                self.components["enable_automatic_video_split"],
                self.components["file_type"]
            ],
            outputs=[
                self.components["dataset_loading"],
                self.components["import_status"]
            ]
        ).success(
            fn=self.app.tabs["import_tab"].on_import_success,
            inputs=[
                self.components["enable_automatic_video_split"],
                self.components["enable_automatic_content_captioning"],
                self.app.tabs["caption_tab"].components["custom_prompt_prefix"]
            ],
            outputs=[
                self.app.tabs_component,
                self.app.tabs["split_tab"].components["video_list"],
                self.app.tabs["split_tab"].components["detect_status"]
            ]
        )

    def set_file_type_and_return(self):
        """Set file type to video and return it"""
        return "video"
    
    def set_file_type_and_return_webdataset(self):
        """Set file type to webdataset and return it"""
        return "webdataset"
    
    def search_datasets(self, query: str):
        """Search datasets on the Hub matching the query"""
        try:
            logger.info(f"Searching for datasets with query: '{query}'")
            results = self.app.importer.search_datasets(query)
            return results, gr.update(visible=True)
        except Exception as e:
            logger.error(f"Error searching datasets: {str(e)}", exc_info=True)
            return [[f"Error: {str(e)}", "", ""]], gr.update(visible=True)
    
    def display_dataset_info(self, evt: gr.SelectData):
        """Display detailed information about the selected dataset"""
        try:
            if not evt or not evt.value:
                logger.warning("No dataset selected in display_dataset_info")
                return (
                    "No dataset selected",  # dataset_info
                    None,                   # dataset_id
                    gr.update(visible=False), # files_section
                    gr.update(visible=False), # video_files_row
                    "",                     # video_count_text
                    gr.update(visible=False), # webdataset_files_row
                    ""                      # webdataset_count_text
                )
            
            dataset_id = evt.value[0] if isinstance(evt.value, list) else evt.value
            logger.info(f"Getting dataset info for: {dataset_id}")
            
            # Use the importer service to get dataset info
            info_text, file_counts, _ = self.app.importer.get_dataset_info(dataset_id)
            
            # Get counts of each file type
            video_count = file_counts.get("video", 0)
            webdataset_count = file_counts.get("webdataset", 0)
            
            # Return all the required outputs individually
            return (
                info_text,                                # dataset_info
                dataset_id,                              # dataset_id
                gr.update(visible=True),                 # files_section
                gr.update(visible=video_count > 0),      # video_files_row
                f"Contains {video_count} video file{'s' if video_count != 1 else ''}", # video_count_text
                gr.update(visible=webdataset_count > 0), # webdataset_files_row
                f"Contains {webdataset_count} WebDataset (.tar) file{'s' if webdataset_count != 1 else ''}" # webdataset_count_text
            )
        except Exception as e:
            logger.error(f"Error displaying dataset info: {str(e)}", exc_info=True)
            return (
                f"Error loading dataset information: {str(e)}", # dataset_info
                None,                                          # dataset_id
                gr.update(visible=False),                      # files_section
                gr.update(visible=False),                      # video_files_row
                "",                                            # video_count_text
                gr.update(visible=False),                      # webdataset_files_row
                ""                                             # webdataset_count_text
            )
    
    def download_file_group(self, dataset_id: str, enable_splitting: bool, file_type: str) -> Tuple[gr.update, str]:
        """Handle download of a group of files (videos or WebDatasets)"""
        try:
            if not dataset_id:
                return gr.update(visible=False), "No dataset selected"
            
            logger.info(f"Starting download of {file_type} files from dataset: {dataset_id}")
            
            # Show loading indicator
            loading_msg = gr.update(
                value=f"## Downloading {file_type} files from {dataset_id}\n\nThis may take some time...",
                visible=True
            )
            status_msg = f"Downloading {file_type} files from {dataset_id}..."
            
            # Use the async version in a non-blocking way
            asyncio.create_task(self._download_file_group_bg(dataset_id, file_type, enable_splitting))
            
            return loading_msg, status_msg
            
        except Exception as e:
            error_msg = f"Error initiating download: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return gr.update(visible=False), error_msg
    
    async def _download_file_group_bg(self, dataset_id: str, file_type: str, enable_splitting: bool):
        """Background task for group file download"""
        try:
            # This will execute in the background
            await self.app.importer.download_file_group(dataset_id, file_type, enable_splitting)
        except Exception as e:
            logger.error(f"Error in background file group download: {str(e)}", exc_info=True)