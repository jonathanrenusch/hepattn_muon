import os

from lightning.pytorch.loggers import CometLogger


class MyCometLogger(CometLogger):
    """Wrap CometLogger to fix issues with CLI arguments.

    Overrides ``save_dir`` so that ``trainer.log_dir`` resolves to a
    per-experiment directory::

        <offline_directory>/<experiment_key>/

    This ensures checkpoints, configs, and other artefacts are cleanly
    separated by experiment rather than all landing in one shared folder.
    """

    def __init__(
        self,
        name: str,
        project_name: str = "colliderml-track-regression",
        offline_directory: str | None = None,
        log_env_details: bool = True,
        **kwargs,
    ):
        assert offline_directory is not None, "offline_directory must be specified for MyCometLogger"
        self._offline_directory = offline_directory
        super().__init__(
            name=name,
            project_name=project_name,
            offline_directory=offline_directory,
            log_env_details=log_env_details,
            **kwargs,
        )

    @property
    def save_dir(self) -> str:
        """Return a per-experiment directory using the Comet experiment key.

        Lightning's ``Trainer.log_dir`` uses ``logger.save_dir`` for
        non-TensorBoard loggers, so placing the experiment key here gives
        each run its own directory for checkpoints and metadata.
        """
        key = self.version  # experiment key (hex string)
        if key is not None:
            return os.path.join(self._offline_directory, key)
        return self._offline_directory
