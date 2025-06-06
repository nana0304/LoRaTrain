from sd_scripts.library import train_util


class CustomLogger:
    def __init__(self, args):
        import wandb
        self.wandb = wandb
        self.accumulation = args.gradient_accumulation_steps
        self.args = args
        self.loss_sum = 0.0
        self.loss_count = 0

    @property
    def accelerator(self):
        return self._accelerator

    @accelerator.setter
    def accelerator(self, value):
        self._accelerator = value
    
    def _initialize_tracker(self):
        # Ensure WandB tracker is initialized
        if self.accelerator.get_tracker("wandb"):
            self.wandb_tracker = self.accelerator.get_tracker("wandb")
        elif self.wandb.run is None:
            init_kwargs = {}
            if self.args.wandb_run_name:
                init_kwargs["wandb"] = {"name": self.args.wandb_run_name}
            if self.args.log_tracker_config is not None:
                import toml
                init_kwargs = toml.load(self.args.log_tracker_config)

            self.accelerator.init_trackers(
                "network_train" if self.args.log_tracker_name is None else self.args.log_tracker_name,
                config=train_util.get_sanitized_config_or_none(self.args),
                init_kwargs=init_kwargs,
            )
        else:
            raise RuntimeError("Failed to initialize WandB tracker.")

    def log(self, loss, global_step):

        self._initialize_tracker()

        self.loss_sum += loss
        self.loss_count += 1

        if global_step % 10 == 0:
            raw_avg_loss = self.loss_sum / self.loss_count
            effective_step = global_step * self.accumulation

            self.wandb.log({
                "current_loss": loss,
                "raw_average_loss": raw_avg_loss,
                "global_step": global_step,
                "effective_step": effective_step
            }, step=effective_step)

            self.loss_sum = 0.0
            self.loss_count = 0

    def log_named(self, name, value, global_step):

        self._initialize_tracker()

        effective_step = global_step * self.accumulation
        self.wandb.log({
            name: value,
            f"{name}_global_step": global_step,
            f"{name}_effective_step": effective_step
        }, step=effective_step)
