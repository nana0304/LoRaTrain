class CustomLogger:
    def __init__(self, args):
        import wandb
        self.wandb = wandb
        self.accumulation = args.gradient_accumulation_steps
        self.args = args
        self.loss_sum = 0.0
        self.loss_count = 0
        self.alpha = 0.1  # Smoothing factor for moving average
        self.defined_metrics = set()
        self.named_moving_avg = {}  # Dictionary to store moving averages for named values

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

            # Lazy import train_util to avoid circular import
            from sd_scripts.library import train_util
            
            self.accelerator.init_trackers(
                "network_train" if self.args.log_tracker_name is None else self.args.log_tracker_name,
                config=train_util.get_sanitized_config_or_none(self.args),
                init_kwargs=init_kwargs,
            )
        else:
            raise RuntimeError("Failed to initialize WandB tracker.")

    def _define_metric(self, name, step_metric="effective_step"):
        if name not in self.defined_metrics:
            self.wandb.define_metric(name, step_metric=step_metric)
            self.defined_metrics.add(name)

    def _calculate_moving_average(self, name, value):
        """Calculate the moving average for a given name and value."""
        if name not in self.named_moving_avg:
            self.named_moving_avg[name] = value  # Initialize on first step
        else:
            self.named_moving_avg[name] = self.alpha * value + (1 - self.alpha) * self.named_moving_avg[name]
        return self.named_moving_avg[name]

    def log(self, loss, global_step):

        self._initialize_tracker()
        self._define_metric("loss/current_loss", step_metric="effective_step")

        self.wandb.log({
            "loss/current_loss": loss,
            "effective_step": global_step * self.accumulation
        }, step=global_step)

        moving_avg_loss = self._calculate_moving_average("loss", loss)

        effective_step = global_step * self.accumulation
        self._define_metric("loss/moving_average_loss", step_metric="effective_step")

        self.wandb.log({
            "loss/moving_average_loss": moving_avg_loss,
            "effective_step": effective_step
        }, step=global_step)

        self.loss_sum += loss
        self.loss_count += 1

        if global_step % 5 == 0:
            raw_avg_loss = self.loss_sum / self.loss_count
            effective_step = global_step * self.accumulation
            self._define_metric("loss/raw_average_loss", step_metric="effective_step")

            self.wandb.log({
                "loss/raw_average_loss": raw_avg_loss,
                "effective_step": effective_step
            }, step=global_step)

            self.loss_sum = 0.0
            self.loss_count = 0

    def log_named(self, name, value, global_step):

        self._initialize_tracker()
        moving_avg_value = self._calculate_moving_average(name, value)

        effective_step = global_step * self.accumulation
        self._define_metric(name, step_metric="effective_step")

        # Log the moving average value with global_step as the step
        self.wandb.log({
            name: moving_avg_value,
            "effective_step": effective_step
        }, step=global_step)

