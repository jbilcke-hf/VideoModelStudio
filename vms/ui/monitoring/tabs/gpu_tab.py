"""
GPU monitoring tab for Video Model Studio UI.
Displays detailed GPU metrics and visualizations.
"""

import gradio as gr
import time
import logging
from pathlib import Path
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from vms.utils.base_tab import BaseTab
from vms.ui.monitoring.utils import human_readable_size

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GPUTab(BaseTab):
    """Tab for GPU-specific monitoring and statistics"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "GPU_tab"
        self.title = "GPU Stats"
        self.refresh_interval = 5
        self.selected_gpu = 0
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the GPU tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                gr.Markdown("## 🖥️ GPU Monitoring")
            
            # No GPUs available message (hidden by default)
            with gr.Row(visible=not self.app.monitoring.gpu.has_nvidia_gpus):
                with gr.Column():
                    gr.Markdown("### No NVIDIA GPUs detected")
                    gr.Markdown("GPU monitoring is only available for NVIDIA GPUs. If you have NVIDIA GPUs installed, ensure the drivers are properly configured.")
            
            # GPU content (only visible if GPUs are available)
            with gr.Row(visible=self.app.monitoring.gpu.has_nvidia_gpus):
                # GPU selector if multiple GPUs
                if self.app.monitoring.gpu.gpu_count > 1:
                    with gr.Column(scale=1):
                        gpu_options = [f"GPU {i}" for i in range(self.app.monitoring.gpu.gpu_count)]
                        self.components["gpu_selector"] = gr.Dropdown(
                            choices=gpu_options,
                            value=gpu_options[0] if gpu_options else None,
                            label="Select GPU",
                            interactive=True
                        )
                
                # Current metrics
                with gr.Column(scale=3):
                    self.components["current_metrics"] = gr.Markdown("Loading GPU metrics...")
            
            # Display GPU metrics in tabs
            with gr.Tabs(visible=self.app.monitoring.gpu.has_nvidia_gpus) as metrics_tabs:
                with gr.Tab(label="Utilization") as util_tab:
                    self.components["utilization_plot"] = gr.Plot()
                
                with gr.Tab(label="Memory") as memory_tab:
                    self.components["memory_plot"] = gr.Plot()
                
                with gr.Tab(label="Power") as power_tab:
                    self.components["power_plot"] = gr.Plot()
            
            # Process information
            with gr.Row(visible=self.app.monitoring.gpu.has_nvidia_gpus):
                with gr.Column():
                    gr.Markdown("### Active Processes")
                    self.components["process_info"] = gr.Markdown("Loading process information...")
            
            # GPU information summary
            with gr.Row(visible=self.app.monitoring.gpu.has_nvidia_gpus):
                with gr.Column():
                    gr.Markdown("### GPU Information")
                    self.components["gpu_info"] = gr.Markdown("Loading GPU information...")
            
            # Toggle for enabling/disabling auto-refresh
            with gr.Row():
                self.components["auto_refresh"] = gr.Checkbox(
                    label=f"Auto refresh (every {self.refresh_interval} seconds)",
                    value=True,
                    info="Automatically refresh GPU metrics"
                )
                self.components["refresh_btn"] = gr.Button("Refresh Now")
            
            # Timer for auto-refresh
            self.components["refresh_timer"] = gr.Timer(
                value=self.refresh_interval
            )
        
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # GPU selector (if multiple GPUs)
        if self.app.monitoring.gpu.gpu_count > 1 and "gpu_selector" in self.components:
            self.components["gpu_selector"].change(
                fn=self.update_selected_gpu,
                inputs=[self.components["gpu_selector"]],
                outputs=[
                    self.components["current_metrics"],
                    self.components["utilization_plot"],
                    self.components["memory_plot"],
                    self.components["power_plot"],
                    self.components["process_info"],
                    self.components["gpu_info"]
                ]
            )
        
        # Manual refresh button
        self.components["refresh_btn"].click(
            fn=self.refresh_all,
            outputs=[
                self.components["current_metrics"],
                self.components["utilization_plot"],
                self.components["memory_plot"],
                self.components["power_plot"],
                self.components["process_info"],
                self.components["gpu_info"]
            ]
        )
        
        # Auto-refresh timer
        self.components["refresh_timer"].tick(
            fn=self.conditional_refresh,
            inputs=[self.components["auto_refresh"]],
            outputs=[
                self.components["current_metrics"],
                self.components["utilization_plot"],
                self.components["memory_plot"],
                self.components["power_plot"],
                self.components["process_info"],
                self.components["gpu_info"]
            ]
        )
    
    def on_enter(self):
        """Called when the tab is selected"""
        # Trigger initial refresh
        return self.refresh_all()
    
    def update_selected_gpu(self, gpu_selector: str) -> Tuple:
        """Update the selected GPU and refresh data
        
        Args:
            gpu_selector: Selected GPU string ("GPU X")
            
        Returns:
            Updated components
        """
        # Extract GPU index from selector string
        try:
            self.selected_gpu = int(gpu_selector.replace("GPU ", ""))
        except (ValueError, AttributeError):
            self.selected_gpu = 0
        
        # Refresh all components with the new selected GPU
        return self.refresh_all()
    
    def conditional_refresh(self, auto_refresh: bool) -> Tuple:
        """Only refresh if auto-refresh is enabled
        
        Args:
            auto_refresh: Whether auto-refresh is enabled
            
        Returns:
            Updated components or unchanged components
        """
        if auto_refresh:
            return self.refresh_all()
        
        # Return current values unchanged if auto-refresh is disabled
        return (
            self.components["current_metrics"].value,
            self.components["utilization_plot"].value,
            self.components["memory_plot"].value,
            self.components["power_plot"].value,
            self.components["process_info"].value,
            self.components["gpu_info"].value
        )
    
    def refresh_all(self) -> Tuple:
        """Refresh all GPU monitoring components
        
        Returns:
            Updated values for all components
        """
        try:
            if not self.app.monitoring.gpu.has_nvidia_gpus:
                return (
                    "No NVIDIA GPUs detected",
                    None,
                    None,
                    None,
                    "No process information available",
                    "No GPU information available"
                )
            
            # Get current metrics for the selected GPU
            all_metrics = self.app.monitoring.gpu.get_current_metrics()
            if not all_metrics or self.selected_gpu >= len(all_metrics):
                return (
                    "GPU metrics not available",
                    None,
                    None,
                    None,
                    "No process information available",
                    "No GPU information available"
                )
            
            # Get selected GPU metrics
            gpu_metrics = all_metrics[self.selected_gpu]
            
            # Format current metrics as markdown
            metrics_html = self.format_current_metrics(gpu_metrics)
            
            # Format process information
            process_info_html = self.format_process_info(gpu_metrics)
            
            # Format GPU information
            gpu_info = self.app.monitoring.gpu.get_gpu_info()
            gpu_info_html = self.format_gpu_info(gpu_info[self.selected_gpu] if self.selected_gpu < len(gpu_info) else {})
            
            # Generate plots
            utilization_plot = self.app.monitoring.gpu.generate_utilization_plot(self.selected_gpu)
            memory_plot = self.app.monitoring.gpu.generate_memory_plot(self.selected_gpu)
            power_plot = self.app.monitoring.gpu.generate_power_plot(self.selected_gpu)
            
            return (
                metrics_html,
                utilization_plot,
                memory_plot,
                power_plot,
                process_info_html,
                gpu_info_html
            )
            
        except Exception as e:
            logger.error(f"Error refreshing GPU data: {str(e)}", exc_info=True)
            error_msg = f"Error retrieving GPU data: {str(e)}"
            return (
                error_msg,
                None,
                None,
                None,
                error_msg,
                error_msg
            )
    
    def format_current_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format current GPU metrics as HTML/Markdown
        
        Args:
            metrics: Current metrics dictionary
            
        Returns:
            Formatted HTML/Markdown string
        """
        if 'error' in metrics:
            return f"Error retrieving GPU metrics: {metrics['error']}"
        
        # Format timestamp
        if isinstance(metrics.get('timestamp'), datetime):
            timestamp_str = metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = "Unknown"
        
        # Style for GPU utilization
        util_style = "color: green;"
        if metrics.get('utilization_gpu', 0) > 90:
            util_style = "color: red; font-weight: bold;"
        elif metrics.get('utilization_gpu', 0) > 70:
            util_style = "color: orange;"
        
        # Style for memory usage
        mem_style = "color: green;"
        if metrics.get('memory_percent', 0) > 90:
            mem_style = "color: red; font-weight: bold;"
        elif metrics.get('memory_percent', 0) > 70:
            mem_style = "color: orange;"
        
        # Style for temperature
        temp_style = "color: green;"
        temp = metrics.get('temperature', 0)
        if temp > 85:
            temp_style = "color: red; font-weight: bold;"
        elif temp > 75:
            temp_style = "color: orange;"
        
        # Memory usage in GB
        memory_used_gb = metrics.get('memory_used', 0) / (1024**3)
        memory_total_gb = metrics.get('memory_total', 0) / (1024**3)
        
        # Power usage and limit
        power_html = ""
        if metrics.get('power_usage') is not None:
            power_html = f"**Power Usage:** {metrics['power_usage']:.1f}W\n"
        
        html = f"""
### Current Status as of {timestamp_str}

**GPU Utilization:** <span style="{util_style}">{metrics.get('utilization_gpu', 0):.1f}%</span>  
**Memory Usage:** <span style="{mem_style}">{metrics.get('memory_percent', 0):.1f}% ({memory_used_gb:.2f}/{memory_total_gb:.2f} GB)</span>  
**Temperature:** <span style="{temp_style}">{metrics.get('temperature', 0)}°C</span>  
{power_html}
"""
        return html
    def format_process_info(self, metrics: Dict[str, Any]) -> str:
        """Format GPU process information as HTML/Markdown
        
        Args:
            metrics: Current metrics dictionary with process information
            
        Returns:
            Formatted HTML/Markdown string
        """
        if 'error' in metrics:
            return "Process information not available"
            
        processes = metrics.get('processes', [])
        if not processes:
            return "No active processes using this GPU"
            
        # Sort processes by memory usage (descending)
        sorted_processes = sorted(processes, key=lambda p: p.get('memory_used', 0), reverse=True)
        
        html = "| PID | Process Name | Memory Usage |\n"
        html += "|-----|-------------|-------------|\n"
        
        for proc in sorted_processes:
            pid = proc.get('pid', 'Unknown')
            name = proc.get('name', 'Unknown')
            mem_mb = proc.get('memory_used', 0) / (1024**2)  # Convert to MB
            
            html += f"| {pid} | {name} | {mem_mb:.1f} MB |\n"
            
        return html
    
    def format_gpu_info(self, info: Dict[str, Any]) -> str:
        """Format GPU information as HTML/Markdown
        
        Args:
            info: GPU information dictionary
            
        Returns:
            Formatted HTML/Markdown string
        """
        if 'error' in info:
            return f"GPU information not available: {info.get('error', 'Unknown error')}"
            
        # Format memory in GB
        memory_total_gb = info.get('memory_total', 0) / (1024**3)
        
        html = f"""
**Name:** {info.get('name', 'Unknown')}  
**Memory:** {memory_total_gb:.2f} GB  
**UUID:** {info.get('uuid', 'N/A')}  
**Compute Capability:** {info.get('compute_capability', 'N/A')}
"""

        # Add power limit if available
        if info.get('power_limit') is not None:
            html += f"**Power Limit:** {info['power_limit']:.1f}W\n"
            
        return html