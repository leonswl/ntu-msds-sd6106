from rich.console import Console

from transformers import (
    TrainerCallback,
    TrainerControl, 
    TrainerState
)

console = Console()

# class EarlyStoppingCallback(TrainerCallback):
#     def __init__(self, patience=3, threshold=0.001):
#         self.patience = patience
#         self.threshold = threshold
#         self.best_loss = float('inf')
#         self.no_improvement_count = 0

#     def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
#         if state.log_history:
#             current_loss = state.log_history[-1]['eval_loss']
#             if current_loss < self.best_loss - self.threshold:
#                 self.best_loss = current_loss
#                 self.no_improvement_count = 0
#             else:
#                 self.no_improvement_count += 1
            
#             if self.no_improvement_count >= self.patience:
#                 control.should_training_stop = True
#                 print(f"Early stopping triggered after {state.global_step} steps")

class BatchSizeCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if args.train_batch_size != args.per_device_train_batch_size:
            console.log(f"Actual batch size: {args.train_batch_size}")

class MetricsLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            console.log(f"Evaluation metrics: {metrics}")
        else:
            console.log("No evaluation metrics available")

