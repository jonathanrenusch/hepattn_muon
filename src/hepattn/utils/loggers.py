from pathlib import Path
from datetime import datetime
from lightning.pytorch.loggers import CometLogger


class MyCometLogger(CometLogger):
    """Wrap CometLogger to fix issues with CLI arguments and automatically log config files."""

    def __init__(self, name: str, offline_directory: str | None = None, log_env_details: bool = True, **kwargs):
        assert offline_directory is not None, "offline_directory must be specified for MyCometLogger"
        
        # Create timestamped directory name but let CometLogger handle the actual creation
        timestamp = datetime.now().strftime("%Y%m%d-T%H%M%S")
        timestamped_name = f"{name}_{timestamp}"
        
        super().__init__(
            name=timestamped_name,
            offline_directory=offline_directory, 
            log_env_details=log_env_details, 
            **kwargs
        )
        self._config_logged = False
        # Store the actual log directory that CometLogger creates
        self._timestamped_dir = Path(self.log_dir) if hasattr(self, 'log_dir') else None

    @property
    def log_dir(self) -> str:
        """Return the log directory path from parent CometLogger."""
        return super().log_dir

    @property 
    def save_dir(self) -> str:
        """Return the log directory path for compatibility."""
        return super().save_dir

    def log_hyperparams(self, params, *args, **kwargs):
        """Override to log config files when hyperparameters are logged."""
        super().log_hyperparams(params, *args, **kwargs)
        
        # Log config files as text if not already done
        if not self._config_logged:
            self._log_config_files()
            self._config_logged = True
    
    def _log_config_files(self):
        """Log all YAML config files in the save directory as text to CometML."""
        try:
            save_dir = Path(self.save_dir)
            if save_dir.exists():
                # Find all YAML config files
                yaml_files = list(save_dir.glob("*.yaml")) + list(save_dir.glob("*.yml"))
                
                for yaml_file in yaml_files:
                    try:
                        # Read and log the config file content as text
                        config_text = yaml_file.read_text(encoding='utf-8')
                        self.experiment.log_text(
                            text=config_text,
                            metadata={"filename": yaml_file.name, "type": "config"}
                        )
                        print(f"Logged config file as text: {yaml_file.name}")
                    except Exception as e:
                        print(f"Failed to log config file {yaml_file.name} as text: {e}")
        except Exception as e:
            print(f"Failed to log config files: {e}")
