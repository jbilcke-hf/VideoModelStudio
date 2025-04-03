"""
System monitoring service for Video Model Studio.
Tracks system resources like CPU, memory, and other metrics.
"""

import os
import time
import logging
import platform
import threading
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import psutil

from vms.ui.monitoring.services.gpu import GPUMonitoringService

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        
        # Initialize GPU monitoring service
        self.gpu = GPUMonitoringService(history_minutes=history_minutes, sample_interval=sample_interval)
        
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

        # Start GPU monitoring if available
        self.gpu.start_monitoring()
        
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

        # Stop GPU monitoring
        self.gpu.stop_monitoring()

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
    
    def get_cpu_data(self) -> pd.DataFrame:
        """Get CPU usage data as a DataFrame
        
        Returns:
            DataFrame with CPU usage data
        """
        if not self.timestamps:
            return pd.DataFrame({
            'time': list(),
            'CPU Usage (%)': list()
        })
            
        data = {
            'time': list(self.timestamps),
            'CPU Usage (%)': list(self.cpu_percent)
        }
        
        # Add temperature if available
        if self.cpu_temp and len(self.cpu_temp) > 0:
            # Ensure temperature data aligns with timestamps
            # If fewer temperature readings than timestamps, pad with None
            temp_data = list(self.cpu_temp)
            if len(temp_data) < len(self.timestamps):
                padding = [None] * (len(self.timestamps) - len(temp_data))
                temp_data = padding + temp_data
            data['CPU Temperature (°C)'] = temp_data
            
        return pd.DataFrame(data)
    
    def get_memory_data(self) -> pd.DataFrame:
        """Get memory usage data as a DataFrame
        
        Returns:
            DataFrame with memory usage data
        """
        if not self.timestamps:
            return pd.DataFrame({
            'time': list(),
            'Memory Usage (%)': list(),
            'Memory Used (GB)': list(),
            'Memory Available (GB)': list()
        })
            
        return pd.DataFrame({
            'time': list(self.timestamps),
            'Memory Usage (%)': list(self.memory_percent),
            'Memory Used (GB)': list(self.memory_used),
            'Memory Available (GB)': list(self.memory_available)
        })
    
    def get_per_core_data(self) -> Dict[int, pd.DataFrame]:
        """Get per-core CPU usage data as DataFrames
        
        Returns:
            Dictionary of DataFrames with per-core CPU usage data
        """
        if not self.timestamps or not self.cpu_cores_percent:
            return {}
            
        core_data = {}
        for core_id, percentages in self.cpu_cores_percent.items():
            # Ensure we don't have more data points than timestamps
            data_length = min(len(percentages), len(self.timestamps))
            core_data[core_id] = pd.DataFrame({
                'time': list(self.timestamps)[-data_length:],
                f'Core {core_id} Usage (%)': list(percentages)[-data_length:]
            })
            
        return core_data
        
    # Replace matplotlib methods with DataFrame methods
    
    # This method is kept for backward compatibility but returns a DataFrame
    def generate_cpu_plot(self) -> pd.DataFrame:
        """Get CPU usage data for plotting
        
        Returns:
            DataFrame with CPU usage data
        """
        return self.get_cpu_data()
    
    # This method is kept for backward compatibility but returns a DataFrame
    def generate_memory_plot(self) -> pd.DataFrame:
        """Get memory usage data for plotting
        
        Returns:
            DataFrame with memory usage data
        """
        return self.get_memory_data()
    
    # This method is kept for backward compatibility but returns a DataFrame of all cores
    def generate_per_core_plot(self) -> pd.DataFrame:
        """Get per-core CPU usage data for plotting
        
        Returns:
            Combined DataFrame with all cores' usage data
        """
        core_data = self.get_per_core_data()
        if not core_data:
            return pd.DataFrame()
            
        # Combine all core data into a single DataFrame using the first core's timestamps
        first_core_id = list(core_data.keys())[0]
        combined_df = core_data[first_core_id][['time']].copy()
        
        for core_id, df in core_data.items():
            combined_df[f'Core {core_id} Usage (%)'] = df[f'Core {core_id} Usage (%)']
            
        return combined_df