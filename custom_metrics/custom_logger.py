class CustomLogger:
    def __init__(self, accumulation):
        import wandb
        self.wandb = wandb
        self.accumulation = accumulation
        self.loss_sum = 0.0
        self.loss_count = 0

    def log(self, loss, global_step):
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
        effective_step = global_step * self.accumulation
        self.wandb.log({
            name: value,
            f"{name}_global_step": global_step,
            f"{name}_effective_step": effective_step
        }, step=effective_step)
