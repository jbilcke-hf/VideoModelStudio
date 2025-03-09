"""
System monitoring service for Video Model Studio.
Tracks system resources like CPU, memory, and other metrics.
"""

import os
import time
import logging
import platform
import threading
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import psutil

# Force the use of the Agg backend which is thread-safe
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

import numpy as np

logger = logging.getLogger(__name__)

class MonitoringService:
    """Service for monitoring system resources and performance"""
    
    def __init__(self, history_minutes: int = 10, sample_interval: int = 5):
        """Initialize the monitoring service
        
        Args:
            history_minutes: How many minutes of history to keep
            sample_interval: How many seconds between samples
        """
        self.history_minutes = history_minutes
        self.sample_interval = sample_interval
        self.max_samples = (history_minutes * 60) // sample_interval
        
        # Initialize data structures for metrics
        self.timestamps = deque(maxlen=self.max_samples)
        self.cpu_percent = deque(maxlen=self.max_samples)
        self.memory_percent = deque(maxlen=self.max_samples)
        self.memory_used = deque(maxlen=self.max_samples)
        self.memory_available = deque(maxlen=self.max_samples)
        
        # CPU temperature history (might not be available on all systems)
        self.cpu_temp = deque(maxlen=self.max_samples)
        
        # Per-core CPU history
        self.cpu_cores_percent = {}
        
        # Track if the monitoring thread is running
        self.is_running = False
        self.thread = None
        
        # Initialize with current values
        self.collect_metrics()
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics
        
        Returns:
            Dictionary of current metrics
        """
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used': psutil.virtual_memory().used / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
            'cpu_temp': None,
            'per_cpu_percent': psutil.cpu_percent(interval=0.1, percpu=True)
        }
        
        # Try to get CPU temperature (platform specific)
        try:
            if platform.system() == 'Linux':
                # Try to get temperature from psutil
                temps = psutil.sensors_temperatures()
                for name, entries in temps.items():
                    if name.startswith(('coretemp', 'k10temp', 'cpu_thermal')):
                        metrics['cpu_temp'] = entries[0].current
                        break
            elif platform.system() == 'Darwin':  # macOS
                # On macOS, we could use SMC reader but it requires additional dependencies
                # Leaving as None for now
                pass
            elif platform.system() == 'Windows':
                # Windows might require WMI, leaving as None for simplicity
                pass
        except (AttributeError, KeyError, IndexError, NotImplementedError):
            # Sensors not available
            pass
        
        return metrics
    
    def update_history(self, metrics: Dict[str, Any]) -> None:
        """Update metric history with new values
        
        Args:
            metrics: New metrics to add to history
        """
        self.timestamps.append(metrics['timestamp'])
        self.cpu_percent.append(metrics['cpu_percent'])
        self.memory_percent.append(metrics['memory_percent'])
        self.memory_used.append(metrics['memory_used'])
        self.memory_available.append(metrics['memory_available'])
        
        if metrics['cpu_temp'] is not None:
            self.cpu_temp.append(metrics['cpu_temp'])
        
        # Update per-core CPU metrics
        for i, percent in enumerate(metrics['per_cpu_percent']):
            if i not in self.cpu_cores_percent:
                self.cpu_cores_percent[i] = deque(maxlen=self.max_samples)
            self.cpu_cores_percent[i].append(percent)
    
    def start_monitoring(self) -> None:
        """Start background thread for collecting metrics"""
        if self.is_running:
            logger.warning("Monitoring thread already running")
            return
            
        self.is_running = True
        
        def _monitor_loop():
            while self.is_running:
                try:
                    metrics = self.collect_metrics()
                    self.update_history(metrics)
                    time.sleep(self.sample_interval)
                except Exception as e:
                    logger.error(f"Error in monitoring thread: {str(e)}", exc_info=True)
                    time.sleep(self.sample_interval)
        
        self.thread = threading.Thread(target=_monitor_loop, daemon=True)
        self.thread.start()
        logger.info("System monitoring thread started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            logger.info("System monitoring thread stopped")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics
        
        Returns:
            Dictionary with current system metrics
        """
        return self.collect_metrics()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get general system information
        
        Returns:
            Dictionary with system details
        """
        cpu_info = {
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'current_frequency': None,
            'architecture': platform.machine(),
        }
        
        # Try to get CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info['current_frequency'] = cpu_freq.current
        except Exception:
            pass
            
        memory_info = {
            'total': psutil.virtual_memory().total / (1024**3),  # GB
            'available': psutil.virtual_memory().available / (1024**3),  # GB
            'used': psutil.virtual_memory().used / (1024**3),  # GB
            'percent': psutil.virtual_memory().percent
        }
        
        disk_info = {}
        for part in psutil.disk_partitions(all=False):
            if os.name == 'nt' and ('cdrom' in part.opts or part.fstype == ''):
                # Skip CD-ROM drives on Windows
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
                disk_info[part.mountpoint] = {
                    'total': usage.total / (1024**3),  # GB
                    'used': usage.used / (1024**3),  # GB
                    'free': usage.free / (1024**3),  # GB
                    'percent': usage.percent
                }
            except PermissionError:
                continue
        
        sys_info = {
            'system': platform.system(),
            'version': platform.version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'python_version': platform.python_version(),
            'uptime': time.time() - psutil.boot_time()
        }
        
        return {
            'cpu': cpu_info,
            'memory': memory_info,
            'disk': disk_info,
            'system': sys_info,
        }
    
    def generate_cpu_plot(self) -> plt.Figure:
        """Generate a plot of CPU usage over time
        
        Returns:
            Matplotlib figure with CPU usage plot
        """
        plt.close('all')  # Close all existing figures
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if not self.timestamps:
            ax.set_title("No CPU data available yet")
            return fig
            
        x = [t.strftime('%H:%M:%S') for t in self.timestamps]
        if len(x) > 10:
            # Show fewer x-axis labels for readability
            step = len(x) // 10
            ax.set_xticks(range(0, len(x), step))
            ax.set_xticklabels([x[i] for i in range(0, len(x), step)])
        
        ax.plot(x, list(self.cpu_percent), 'b-', label='CPU Usage %')
        
        if self.cpu_temp and len(self.cpu_temp) > 0:
            # Plot temperature on a secondary y-axis if available
            ax2 = ax.twinx()
            ax2.plot(x[:len(self.cpu_temp)], list(self.cpu_temp), 'r-', label='CPU Temp °C')
            ax2.set_ylabel('Temperature (°C)', color='r')
            ax2.tick_params(axis='y', colors='r')
            
        ax.set_title('CPU Usage Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Usage %')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        if hasattr(locals(), 'ax2'):
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax.legend(loc='upper left')
            
        plt.tight_layout()
        return fig
    
    def generate_memory_plot(self) -> plt.Figure:
        """Generate a plot of memory usage over time
        
        Returns:
            Matplotlib figure with memory usage plot
        """
        plt.close('all')  # Close all existing figures
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if not self.timestamps:
            ax.set_title("No memory data available yet")
            return fig
            
        x = [t.strftime('%H:%M:%S') for t in self.timestamps]
        if len(x) > 10:
            # Show fewer x-axis labels for readability
            step = len(x) // 10
            ax.set_xticks(range(0, len(x), step))
            ax.set_xticklabels([x[i] for i in range(0, len(x), step)])
        
        ax.plot(x, list(self.memory_percent), 'g-', label='Memory Usage %')
        
        # Add secondary y-axis for absolute memory values
        ax2 = ax.twinx()
        ax2.plot(x, list(self.memory_used), 'm--', label='Used (GB)')
        ax2.plot(x, list(self.memory_available), 'c--', label='Available (GB)')
        ax2.set_ylabel('Memory (GB)')
        
        ax.set_title('Memory Usage Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Usage %')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
            
        plt.tight_layout()
        return fig
    
    def generate_per_core_plot(self) -> plt.Figure:
        """Generate a plot of per-core CPU usage
        
        Returns:
            Matplotlib figure with per-core CPU usage
        """
        num_cores = len(self.cpu_cores_percent)
        if num_cores == 0:
            # No data yet
            plt.close('all')  # Close all existing figures
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title("No per-core CPU data available yet")
            return fig
            
        # Determine grid layout based on number of cores
        if num_cores <= 4:
            rows, cols = 2, 2
        elif num_cores <= 6:
            rows, cols = 2, 3
        elif num_cores <= 9:
            rows, cols = 3, 3
        elif num_cores <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4
            
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        
        x = [t.strftime('%H:%M:%S') for t in self.timestamps]
        if len(x) > 5:
            # Show fewer x-axis labels for readability
            step = len(x) // 5
        else:
            step = 1
            
        for i, (core_id, percentages) in enumerate(self.cpu_cores_percent.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            ax.plot(x[:len(percentages)], list(percentages), 'b-')
            ax.set_title(f'Core {core_id}')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            
            # Add x-axis labels sparingly for readability
            if i >= len(axes) - cols:  # Only for bottom row
                ax.set_xticks(range(0, len(x), step))
                ax.set_xticklabels([x[i] for i in range(0, len(x), step)], rotation=45)
                
        # Hide unused subplots
        for i in range(num_cores, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        return fig