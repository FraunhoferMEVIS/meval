import os

# partially Claude-generated
class Settings:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._init_defaults()
        return cls._instance
    
    def _init_defaults(self):
        # Default settings
        self.debug = self._get_env_bool("MEVAL_DEBUG", False)
        self.N_bootstrap = self._get_env_int("MEVAL_N_BOOTSTRAP", 200)
        self.ci_alpha = self._get_env_float("MEVAL_CI_ALPHA", 0.95)
        self.num_loess_calibration_samples = self._get_env_int("MEVAL_NUM_LOESS_CALIBRATION_SAMPLES", self.N_bootstrap)
        self.add_med_group = self._get_env_bool("MEVAL_ADD_MED_GROUP", False)
        self.ci_plot_alpha = self._get_env_float("MEVAL_CI_PLOT_ALPHA", 0.3)
        self.seed = self._get_env_int("MEVAL_SEED", 49)
        self.parallel = self._get_env_bool("MEVAL_PARALLEL", True if not self.debug else False)
        self.N_test_permut = self._get_env_int("MEVAL_N_TEST_PERMUT", 1000)
        self.max_N_student_bootstrap = self._get_env_int("MEVAL_MAX_N_STUDENT_BOOTSTRAP", 100)

    def update(self, **kwargs) -> None:
        """Update settings with provided values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Unknown setting: {key}")
    
    def reset(self):
        """Reset to default settings."""
        self._init_defaults()
    
    def to_dict(self) -> dict:
        """Export current settings as dictionary."""
        return {
            key: value for key, value in self.__dict__.items() 
            if not key.startswith('_')
        }
    
    def from_dict(self, settings_dict: dict) -> None:
        """Import settings from dictionary."""
        self.update(**settings_dict)

    def _get_env_bool(self, name, default: bool) -> bool:
        """Convert environment variable to boolean."""
        value = os.environ.get(name)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "y", "on")
    
    def _get_env_int(self, name, default: int) -> int:
        """Convert environment variable to integer."""
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default
    
    def _get_env_float(self, name, default: float) -> float:
        """Convert environment variable to float."""
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def load_from_env(self, prefix: str = "MEVAL_") -> None:
        """Load settings from environment variables."""
        import os
        
        # Example mappings between settings and environment variables
        mappings = {
            "debug": (self._get_env_bool, f"{prefix}DEBUG"),
            "N_bootstrap": (self._get_env_int, f"{prefix}N_BOOTSTRAP"),
            "ci_alpha": (self._get_env_float, f"{prefix}CI_ALPHA"),
            "num_loess_calibration_samples": (self._get_env_int, f"{prefix}NUM_LOESS_CALIBRATION_SAMPLES"),
            "add_med_group": (self._get_env_bool, f"{prefix}ADD_MED_GROUP"),
            "ci_plot_alpha": (self._get_env_float, f"{prefix}CI_PLOT_ALPHA"),
            "seed": (self._get_env_int, f"{prefix}SEED"),
            "parallel": (self._get_env_bool, f"{prefix}PARALLEL"),
            "N_test_permut": (self._get_env_int, f"{prefix}N_TEST_PERMUT"),
            "max_N_student_bootstrap": (self._get_env_int, f"{prefix}MAX_N_STUDENT_BOOTSTRAP")
        }
        
        for setting_name, (converter, env_name) in mappings.items():
            if env_name in os.environ:
                current_value = getattr(self, setting_name)
                setattr(self, setting_name, converter(env_name, current_value))

    def __str__(self) -> str:
        """String representation for print(settings)."""
        lines = ["Settings:"]
        settings_dict = self.to_dict()
        
        # Find the longest key name for alignment
        max_key_length = max(len(key) for key in settings_dict.keys()) if settings_dict else 0
        
        # Format each setting
        for key, value in sorted(settings_dict.items()):
            lines.append(f"  {key:<{max_key_length}} = {value!r}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        settings_dict = self.to_dict()
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(settings_dict.items()))
        return f"Settings({items})"                
                
# Create a singleton instance
settings = Settings()