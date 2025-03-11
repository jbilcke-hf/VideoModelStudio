"""
System monitoring tab for Video Model Studio UI.
Displays system metrics like CPU, memory usage, and temperatures.
"""

import gradio as gr
import time
import logging
from pathlib import Path
import os
import psutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from .base_tab import BaseTab
from ..config import STORAGE_PATH

logger = logging.getLogger(__name__)

def get_folder_size(path):
    """Calculate the total size of a folder in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):  # Skip symlinks
                total_size += os.path.getsize(file_path)
    return total_size

def human_readable_size(size_bytes):
    """Convert a size in bytes to a human-readable string"""
    if size_bytes == 0:
        return "0 B"
    size_names = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.2f} {size_names[i]}"

class MonitorTab(BaseTab):
    """Monitor tab for system resource monitoring"""
    
    def __init__(self, app_state):
        super().__init__(app_state)
        self.id = "monitor_tab"
        self.title = "5️⃣  Monitor"
        self.refresh_interval = 8
    
    def create(self, parent=None) -> gr.TabItem:
        """Create the Monitor tab UI components"""
        with gr.TabItem(self.title, id=self.id) as tab:
            with gr.Row():
                gr.Markdown("## System Monitoring")
  
            # Current metrics
            with gr.Row():
                with gr.Column(scale=1):
                    self.components["current_metrics"] = gr.Markdown("Loading current metrics...")
            
            # CPU and Memory charts in tabs
            with gr.Tabs() as metrics_tabs:
                with gr.Tab(label="CPU Usage") as cpu_tab:
                    self.components["cpu_plot"] = gr.Plot()
                
                with gr.Tab(label="Memory Usage") as memory_tab:
                    self.components["memory_plot"] = gr.Plot()
                    
                #with gr.Tab(label="Per-Core CPU") as per_core_tab:
                #    self.components["per_core_plot"] = gr.Plot()

            # System information summary in columns
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### System Information")
                    self.components["system_info"] = gr.Markdown("Loading system information...")
                
                with gr.Column(scale=1):
                    gr.Markdown("### CPU Information")
                    self.components["cpu_info"] = gr.Markdown("Loading CPU information...")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Memory Information")
                    self.components["memory_info"] = gr.Markdown("Loading memory information...")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Storage Information")
                    self.components["storage_info"] = gr.Markdown("Loading storage information...")
          
            # Toggle for enabling/disabling auto-refresh
            with gr.Row():
                self.components["auto_refresh"] = gr.Checkbox(
                    label=f"Auto refresh (every {self.refresh_interval} seconds)",
                    value=True,
                    info="Automatically refresh system metrics"
                )
                self.components["refresh_btn"] = gr.Button("Refresh Now")
            
            # Timer for auto-refresh
            self.components["refresh_timer"] = gr.Timer(
                value=self.refresh_interval
            )
            
        return tab
    
    def connect_events(self) -> None:
        """Connect event handlers to UI components"""
        # Manual refresh button
        self.components["refresh_btn"].click(
            fn=self.refresh_all,
            outputs=[
                self.components["system_info"],
                self.components["cpu_info"],
                self.components["memory_info"],
                self.components["storage_info"],
                self.components["current_metrics"],
                self.components["cpu_plot"],
                self.components["memory_plot"],
                #self.components["per_core_plot"]
            ]
        )
        
        # Auto-refresh timer
        self.components["refresh_timer"].tick(
            fn=self.conditional_refresh,
            inputs=[self.components["auto_refresh"]],
            outputs=[
                self.components["system_info"],
                self.components["cpu_info"],
                self.components["memory_info"],
                self.components["storage_info"],
                self.components["current_metrics"],
                self.components["cpu_plot"],
                self.components["memory_plot"],
                #self.components["per_core_plot"]
            ]
        )
    
    def on_enter(self):
        """Called when the tab is selected"""
        # Start monitoring service if not already running
        if not self.app.monitoring.is_running:
            self.app.monitoring.start_monitoring()
        
        # Trigger initial refresh
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
            self.components["system_info"].value,
            self.components["cpu_info"].value,
            self.components["memory_info"].value,
            self.components["storage_info"].value,
            self.components["current_metrics"].value,
            self.components["cpu_plot"].value,
            self.components["memory_plot"].value, 
            #self.components["per_core_plot"].value
        )
    
    def refresh_all(self) -> Tuple:
        """Refresh all monitoring components
        
        Returns:
            Updated values for all components
        """
        try:
            # Get system info
            system_info = self.app.monitoring.get_system_info()
            
            # Split system info into separate components
            system_info_html = self.format_system_info(system_info)
            cpu_info_html = self.format_cpu_info(system_info)
            memory_info_html = self.format_memory_info(system_info)
            storage_info_html = self.format_storage_info()
            
            # Get current metrics
            # current_metrics = self.app.monitoring.get_current_metrics()
            metrics_html = "" # self.format_current_metrics(current_metrics)
            
            # Generate plots
            cpu_plot = self.app.monitoring.generate_cpu_plot()
            memory_plot = self.app.monitoring.generate_memory_plot()
            #per_core_plot = self.app.monitoring.generate_per_core_plot()
            
            return (
                system_info_html, 
                cpu_info_html, 
                memory_info_html, 
                storage_info_html, 
                metrics_html, 
                cpu_plot, 
                memory_plot, 
                #per_core_plot
            )
            
        except Exception as e:
            logger.error(f"Error refreshing monitoring data: {str(e)}", exc_info=True)
            error_msg = f"Error retrieving data: {str(e)}"
            return (
                error_msg,
                error_msg,
                error_msg,
                error_msg,
                error_msg,
                None,
                None,
                #None
            )
    
    def format_system_info(self, system_info: Dict[str, Any]) -> str:
        """Format system information as HTML
        
        Args:
            system_info: System information dictionary
            
        Returns:
            Formatted HTML string
        """
        sys = system_info['system']
        uptime_str = self.format_uptime(sys['uptime'])
        
        html = f"""
**System:** {sys['system']} ({sys['platform']})  
**Hostname:** {sys['hostname']}  
**Uptime:** {uptime_str}  
**Python Version:** {sys['python_version']}
        """
        return html
    
    def format_cpu_info(self, system_info: Dict[str, Any]) -> str:
        """Format CPU information as HTML
        
        Args:
            system_info: System information dictionary
            
        Returns:
            Formatted HTML string
        """
        cpu = system_info['cpu']
        sys = system_info['system']
        
        # Format CPU frequency
        cpu_freq = "N/A"
        if cpu['current_frequency']:
            cpu_freq = f"{cpu['current_frequency'] / 1000:.2f} GHz"
            
        html = f"""
**Processor:** {sys['processor'] or cpu['architecture']}  
**Physical Cores:** {cpu['cores_physical']}  
**Logical Cores:** {cpu['cores_logical']}  
**Current Frequency:** {cpu_freq}
        """
        return html
    
    def format_memory_info(self, system_info: Dict[str, Any]) -> str:
        """Format memory information as HTML
        
        Args:
            system_info: System information dictionary
            
        Returns:
            Formatted HTML string
        """
        memory = system_info['memory']
        
        html = f"""
**Total Memory:** {memory['total']:.2f} GB  
**Available Memory:** {memory['available']:.2f} GB  
**Used Memory:** {memory['used']:.2f} GB  
**Usage:** {memory['percent']}%
        """
        return html
    
    def format_storage_info(self) -> str:
        """Format storage information as HTML, focused on STORAGE_PATH
        
        Returns:
            Formatted HTML string
        """
        try:
            # Get total size of STORAGE_PATH
            total_size = get_folder_size(STORAGE_PATH)
            total_size_readable = human_readable_size(total_size)
            
            html = f"**Total Storage Used:** {total_size_readable}\n\n"
            
            # Get size of each subfolder
            html += "**Subfolder Sizes:**\n\n"
            
            for subfolder in sorted(STORAGE_PATH.iterdir()):
                if subfolder.is_dir():
                    folder_size = get_folder_size(subfolder)
                    folder_size_readable = human_readable_size(folder_size)
                    percentage = (folder_size / total_size * 100) if total_size > 0 else 0
                    
                    folder_name = subfolder.name
                    html += f"* **{folder_name}**: {folder_size_readable} ({percentage:.1f}%)\n"
            
            return html
            
        except Exception as e:
            logger.error(f"Error getting folder sizes: {str(e)}", exc_info=True)
            return f"Error getting folder sizes: {str(e)}"
    
    def format_current_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format current metrics as HTML
        
        Args:
            metrics: Current metrics dictionary
            
        Returns:
            Formatted HTML string
        """
        timestamp = metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Style for CPU usage
        cpu_style = "color: green;"
        if metrics['cpu_percent'] > 90:
            cpu_style = "color: red; font-weight: bold;"
        elif metrics['cpu_percent'] > 70:
            cpu_style = "color: orange;"
            
        # Style for memory usage
        mem_style = "color: green;"
        if metrics['memory_percent'] > 90:
            mem_style = "color: red; font-weight: bold;"
        elif metrics['memory_percent'] > 70:
            mem_style = "color: orange;"
            
        # Temperature info
        temp_html = ""
        if metrics['cpu_temp'] is not None:
            temp_style = "color: green;"
            if metrics['cpu_temp'] > 80:
                temp_style = "color: red; font-weight: bold;"
            elif metrics['cpu_temp'] > 70:
                temp_style = "color: orange;"
                
            temp_html = f"""
**CPU Temperature:** <span style="{temp_style}">{metrics['cpu_temp']:.1f}°C</span>
"""
            
        html = f"""
**CPU Usage:** <span style="{cpu_style}">{metrics['cpu_percent']:.1f}%</span>
**Memory Usage:** <span style="{mem_style}">{metrics['memory_percent']:.1f}% ({metrics['memory_used']:.2f}/{metrics['memory_available']:.2f} GB)</span>  
{temp_html}
        """
        
        # Add per-CPU core info
        html += "\n"
        
        per_cpu = metrics['per_cpu_percent']
        cols = 4  # 4 cores per row
        
        # Create a grid layout for cores
        for i in range(0, len(per_cpu), cols):
            row_cores = per_cpu[i:i+cols]
            row_html = ""
            
            for j, usage in enumerate(row_cores):
                core_id = i + j
                core_style = "color: green;"
                if usage > 90:
                    core_style = "color: red; font-weight: bold;"
                elif usage > 70:
                    core_style = "color: orange;"
                    
                row_html += f"**Core {core_id}:** <span style='{core_style}'>{usage:.1f}%</span>&nbsp;&nbsp;&nbsp;"
                
            html += row_html + "\n"
            
        return html
    
    def format_uptime(self, seconds: float) -> str:
        """Format uptime in seconds to a human-readable string
        
        Args:
            seconds: Uptime in seconds
            
        Returns:
            Formatted uptime string
        """
        days = int(seconds // 86400)
        seconds %= 86400
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0 or days > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        
        return ", ".join(parts)