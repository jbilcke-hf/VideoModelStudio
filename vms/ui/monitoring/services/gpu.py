"""
GPU monitoring service for Video Model Studio.
Tracks NVIDIA GPU resources like utilization, memory, and temperature.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from datetime import datetime

# Force the use of the Agg backend which is thread-safe
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Optional import of pynvml
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.info("pynvml not available, GPU monitoring will be limited")

class GPUMonitoringService:
    """Service for monitoring NVIDIA GPU resources"""
    
    def __init__(self, history_minutes: int = 10, sample_interval: int = 5):
        """Initialize the GPU monitoring service
        
        Args:
            history_minutes: How many minutes of history to keep
            sample_interval: How many seconds between samples
        """
        self.history_minutes = history_minutes
        self.sample_interval = sample_interval
        self.max_samples = (history_minutes * 60) // sample_interval
        
        # Track if the monitoring thread is running
        self.is_running = False
        self.thread = None
        
        # Check if NVIDIA GPUs are available
        self.has_nvidia_gpus = False
        self.gpu_count = 0
        self.device_info = []
        self.history = {}
        
        # Try to initialize NVML
        self._initialize_nvml()
        
        # Initialize history data structures if GPUs are available
        if self.has_nvidia_gpus:
            self._initialize_history()
    
    def _initialize_nvml(self):
        """Initialize NVIDIA Management Library"""
        if not PYNVML_AVAILABLE:
            logger.info("pynvml module not installed, GPU monitoring disabled")
            return
            
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.has_nvidia_gpus = self.gpu_count > 0
            
            if self.has_nvidia_gpus:
                logger.info(f"Successfully initialized NVML, found {self.gpu_count} GPU(s)")
                # Get static information about each GPU
                for i in range(self.gpu_count):
                    self.device_info.append(self._get_device_info(i))
            else:
                logger.info("No NVIDIA GPUs found")
                
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {str(e)}")
            self.has_nvidia_gpus = False
    
    def _initialize_history(self):
        """Initialize data structures for storing metric history"""
        for i in range(self.gpu_count):
            self.history[i] = {
                'timestamps': deque(maxlen=self.max_samples),
                'utilization': deque(maxlen=self.max_samples),
                'memory_used': deque(maxlen=self.max_samples),
                'memory_total': deque(maxlen=self.max_samples),
                'memory_percent': deque(maxlen=self.max_samples),
                'temperature': deque(maxlen=self.max_samples),
                'power_usage': deque(maxlen=self.max_samples),
                'power_limit': deque(maxlen=self.max_samples),
            }
    
    def _get_device_info(self, device_index: int) -> Dict[str, Any]:
        """Get static information about a GPU device
        
        Args:
            device_index: Index of the GPU device
            
        Returns:
            Dictionary with device information
        """
        if not PYNVML_AVAILABLE or not self.has_nvidia_gpus:
            return {"error": "NVIDIA GPUs not available"}
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            
            # Get device name (decode if it's bytes)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
                
            # Get device UUID
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode('utf-8')
                
            # Get memory info, compute capability
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            
            # Get power limits if available
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # Convert to watts
            except pynvml.NVMLError:
                power_limit = None
                
            return {
                'index': device_index,
                'name': name,
                'uuid': uuid,
                'memory_total': memory_info.total,
                'memory_total_gb': memory_info.total / (1024**3),  # Convert to GB
                'compute_capability': f"{compute_capability[0]}.{compute_capability[1]}",
                'power_limit': power_limit
            }
            
        except Exception as e:
            logger.error(f"Error getting device info for GPU {device_index}: {str(e)}")
            return {"error": str(e), "index": device_index}
    
    def collect_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Collect current GPU metrics for all available GPUs
        
        Returns:
            List of dictionaries with current metrics for each GPU
        """
        if not PYNVML_AVAILABLE or not self.has_nvidia_gpus:
            return []
            
        metrics = []
        timestamp = datetime.now()
        
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get utilization rates
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get memory information
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get temperature
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Get power usage if available
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except pynvml.NVMLError:
                    power_usage = None
                
                # Get process information
                processes = []
                try:
                    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        try:
                            process_name = pynvml.nvmlSystemGetProcessName(proc.pid)
                            if isinstance(process_name, bytes):
                                process_name = process_name.decode('utf-8')
                        except pynvml.NVMLError:
                            process_name = f"Unknown (PID: {proc.pid})"
                            
                        processes.append({
                            'pid': proc.pid,
                            'name': process_name,
                            'memory_used': proc.usedGpuMemory,
                            'memory_used_mb': proc.usedGpuMemory / (1024**2)  # Convert to MB
                        })
                except pynvml.NVMLError:
                    # Unable to get process information, continue with empty list
                    pass
                
                gpu_metrics = {
                    'index': i,
                    'timestamp': timestamp,
                    'utilization_gpu': utilization.gpu,
                    'utilization_memory': utilization.memory,
                    'memory_total': memory_info.total,
                    'memory_used': memory_info.used,
                    'memory_free': memory_info.free,
                    'memory_percent': (memory_info.used / memory_info.total) * 100,
                    'temperature': temperature,
                    'power_usage': power_usage,
                    'processes': processes
                }
                
                metrics.append(gpu_metrics)
                
            except Exception as e:
                logger.error(f"Error collecting metrics for GPU {i}: {str(e)}")
                metrics.append({
                    'index': i,
                    'error': str(e)
                })
        
        return metrics
    
    def update_history(self):
        """Update GPU metrics history"""
        if not self.has_nvidia_gpus:
            return
            
        current_metrics = self.collect_gpu_metrics()
        timestamp = datetime.now()
        
        for gpu_metrics in current_metrics:
            if 'error' in gpu_metrics:
                continue
                
            idx = gpu_metrics['index']
            
            self.history[idx]['timestamps'].append(timestamp)
            self.history[idx]['utilization'].append(gpu_metrics['utilization_gpu'])
            self.history[idx]['memory_used'].append(gpu_metrics['memory_used'])
            self.history[idx]['memory_total'].append(gpu_metrics['memory_total'])
            self.history[idx]['memory_percent'].append(gpu_metrics['memory_percent'])
            self.history[idx]['temperature'].append(gpu_metrics['temperature'])
            
            if gpu_metrics['power_usage'] is not None:
                self.history[idx]['power_usage'].append(gpu_metrics['power_usage'])
            else:
                self.history[idx]['power_usage'].append(0)
                
            # Store power limit in history (static but kept for consistency)
            info = self.device_info[idx]
            if 'power_limit' in info and info['power_limit'] is not None:
                self.history[idx]['power_limit'].append(info['power_limit'])
            else:
                self.history[idx]['power_limit'].append(0)
    
    def start_monitoring(self):
        """Start background thread for collecting GPU metrics"""
        if self.is_running:
            logger.warning("GPU monitoring thread already running")
            return
            
        if not self.has_nvidia_gpus:
            logger.info("No NVIDIA GPUs found, not starting monitoring thread")
            return
            
        import threading
        
        self.is_running = True
        
        def _monitor_loop():
            while self.is_running:
                try:
                    self.update_history()
                    time.sleep(self.sample_interval)
                except Exception as e:
                    logger.error(f"Error in GPU monitoring thread: {str(e)}", exc_info=True)
                    time.sleep(self.sample_interval)
        
        self.thread = threading.Thread(target=_monitor_loop, daemon=True)
        self.thread.start()
        logger.info("GPU monitoring thread started")
    
    def stop_monitoring(self):
        """Stop the GPU monitoring thread"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            logger.info("GPU monitoring thread stopped")
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get information about all available GPUs
        
        Returns:
            List of dictionaries with GPU information
        """
        return self.device_info
    
    def get_current_metrics(self) -> List[Dict[str, Any]]:
        """Get current metrics for all GPUs
        
        Returns:
            List of dictionaries with current GPU metrics
        """
        return self.collect_gpu_metrics()
    
    def generate_utilization_plot(self, gpu_index: int) -> plt.Figure:
        """Generate a plot of GPU utilization over time
        
        Args:
            gpu_index: Index of the GPU to plot
            
        Returns:
            Matplotlib figure with utilization plot
        """
        plt.close('all')  # Close all existing figures
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if not self.has_nvidia_gpus or gpu_index not in self.history:
            ax.set_title(f"No data available for GPU {gpu_index}")
            return fig
            
        history = self.history[gpu_index]
        if not history['timestamps']:
            ax.set_title(f"No history data for GPU {gpu_index}")
            return fig
            
        # Convert timestamps to strings
        x = [t.strftime('%H:%M:%S') for t in history['timestamps']]
        
        # If we have many points, show fewer labels for readability
        if len(x) > 10:
            step = len(x) // 10
            ax.set_xticks(range(0, len(x), step))
            ax.set_xticklabels([x[i] for i in range(0, len(x), step)], rotation=45)
        
        # Plot utilization
        ax.plot(x, list(history['utilization']), 'b-', label='GPU Utilization %')
        ax.set_ylim(0, 100)
        
        # Add temperature on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(x, list(history['temperature']), 'r-', label='Temperature °C')
        ax2.set_ylabel('Temperature (°C)', color='r')
        ax2.tick_params(axis='y', colors='r')
        
        # Set labels and title
        ax.set_title(f'GPU {gpu_index} Utilization Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Utilization %')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def generate_memory_plot(self, gpu_index: int) -> plt.Figure:
        """Generate a plot of GPU memory usage over time
        
        Args:
            gpu_index: Index of the GPU to plot
            
        Returns:
            Matplotlib figure with memory usage plot
        """
        plt.close('all')  # Close all existing figures
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if not self.has_nvidia_gpus or gpu_index not in self.history:
            ax.set_title(f"No data available for GPU {gpu_index}")
            return fig
            
        history = self.history[gpu_index]
        if not history['timestamps']:
            ax.set_title(f"No history data for GPU {gpu_index}")
            return fig
            
        # Convert timestamps to strings
        x = [t.strftime('%H:%M:%S') for t in history['timestamps']]
        
        # If we have many points, show fewer labels for readability
        if len(x) > 10:
            step = len(x) // 10
            ax.set_xticks(range(0, len(x), step))
            ax.set_xticklabels([x[i] for i in range(0, len(x), step)], rotation=45)
        
        # Plot memory percentage
        ax.plot(x, list(history['memory_percent']), 'g-', label='Memory Usage %')
        ax.set_ylim(0, 100)
        
        # Add absolute memory values on secondary y-axis (convert to GB)
        ax2 = ax.twinx()
        memory_used_gb = [m / (1024**3) for m in history['memory_used']]
        memory_total_gb = [m / (1024**3) for m in history['memory_total']]
        
        ax2.plot(x, memory_used_gb, 'm--', label='Used (GB)')
        ax2.set_ylabel('Memory (GB)')
        
        # Set labels and title
        ax.set_title(f'GPU {gpu_index} Memory Usage Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Usage %')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def generate_power_plot(self, gpu_index: int) -> plt.Figure:
        """Generate a plot of GPU power usage over time
        
        Args:
            gpu_index: Index of the GPU to plot
            
        Returns:
            Matplotlib figure with power usage plot
        """
        plt.close('all')  # Close all existing figures
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if not self.has_nvidia_gpus or gpu_index not in self.history:
            ax.set_title(f"No data available for GPU {gpu_index}")
            return fig
            
        history = self.history[gpu_index]
        if not history['timestamps'] or not any(history['power_usage']):
            ax.set_title(f"No power data for GPU {gpu_index}")
            return fig
            
        # Convert timestamps to strings
        x = [t.strftime('%H:%M:%S') for t in history['timestamps']]
        
        # If we have many points, show fewer labels for readability
        if len(x) > 10:
            step = len(x) // 10
            ax.set_xticks(range(0, len(x), step))
            ax.set_xticklabels([x[i] for i in range(0, len(x), step)], rotation=45)
        
        # Plot power usage
        power_usage = list(history['power_usage'])
        if any(power_usage):  # Only plot if we have actual power data
            ax.plot(x, power_usage, 'b-', label='Power Usage (W)')
            
            # Get power limit if available
            power_limit = list(history['power_limit'])
            if any(power_limit):  # Only plot if we have power limit data
                # Show power limit as horizontal line
                limit = max(power_limit)  # Should be constant, but take max just in case
                if limit > 0:
                    ax.axhline(y=limit, color='r', linestyle='--', label=f'Power Limit ({limit}W)')
            
            # Set labels and title
            ax.set_title(f'GPU {gpu_index} Power Usage Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Power (Watts)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
        else:
            ax.set_title(f"Power data not available for GPU {gpu_index}")
        
        plt.tight_layout()
        return fig
        
    def shutdown(self):
        """Clean up resources when shutting down"""
        self.stop_monitoring()
        
        # Shutdown NVML if it was initialized
        if PYNVML_AVAILABLE and self.has_nvidia_gpus:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown complete")
            except Exception as e:
                logger.error(f"Error during NVML shutdown: {str(e)}")