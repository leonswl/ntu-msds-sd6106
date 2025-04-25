"""flowertune-llm: A Flower / FlowerTune app."""

import os
from datetime import datetime, date
from logging import INFO, DEBUG, ERROR
from typing import Dict, Optional, Tuple
import logging

import pandas as pd
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Context, ndarrays_to_parameters, NDArrays, Scalar
from flwr.common.config import unflatten_dict
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from omegaconf import DictConfig
from transformers import AutoTokenizer

from flowertune_llm.models import get_model, get_parameters, set_parameters, compute_server_metrics
from flowertune_llm.dataset import replace_keys, load_data

# Global configuration store
global_cfg = None

# Initialize a global list to accumulate evaluation metrics
evaluation_metrics = []

# os.environ["WANDB_DISABLED"] = "true"

STRATEGIES = [
    "fedavg",
    "fedprox",
    "fedavgm",
    "fedadagrad",
    "fedadam"
]

def evaluate_config(server_round: int):
    global global_cfg  # Access global config
    log(INFO, f"Global Config: {global_cfg}")

    return {"global_cfg": global_cfg}

# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
def get_evaluate_fn(
        model_cfg,
        save_every_round,
        total_round,
        save_path, 
        dataset_name: str,
        batch_size: int
        ):
    """Return an evaluation function for saving global model."""

    # List to collect evaluation metrics
    evaluation_metrics = []

    def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
            ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        log(INFO, f"Received config in evaluate: {config}")

        # Save model
        log(INFO, f"Server Round: {server_round}")
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg)
            set_parameters(model, parameters)

            model.save_pretrained(f"{save_path}/peft_{server_round}")

            tokenizer = AutoTokenizer.from_pretrained(
                model_cfg.name, use_fast=True, padding_side="right"
            )
            tokenizer.pad_token = tokenizer.eos_token

            # Load eval dataset
            _, _, test_set = load_data(dataset_name=dataset_name)

            eval_dataloader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=12
                )

            # Compute evaluation loss
            eval_loss, eval_accuracy = compute_server_metrics(model, tokenizer, eval_dataloader)

            # Save evaluation metric for this round
            evaluation_metrics.append({
                "round": server_round,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy
            })

            # Write metrics to CSV file after each round
            df = pd.DataFrame(evaluation_metrics)
            df.to_csv(os.path.join(save_path, "evaluation_metrics.csv"), index=False)

            log(INFO, f"Aggregate Evaluation SUCCESS\nevaluation_metrics: {evaluation_metrics}")

            return eval_loss, {"evaluation_metrics": evaluation_metrics}
            # return evaluation_metrics, {"evaluation_metrics": evaluation_metrics}
        
        else:
            return 0.0, {}

    return evaluate

def aggregate_evaluation_metrics(results):
    if not results:
        return {}

    # Unpack the results
    total_examples = sum(num_examples for num_examples, _ in results)
    weighted_metrics = {}

    for num_examples, metrics in results:
        weight = num_examples / total_examples
        for metric_name, metric_value in metrics.items():
            if metric_name not in weighted_metrics:
                weighted_metrics[metric_name] = 0
            weighted_metrics[metric_name] += weight * metric_value

    return weighted_metrics


def evaluate_metrics_aggregation_fn(results):
    if not results:
        log(INFO, "No evaluation results received")
        return {}

    # Log failures
    failures = [failure for _, failure in results if failure]
    if failures:
        log(ERROR, f"Evaluation failures: {failures}")

    # Aggregate successful results
    successful_results = [(num_examples, metrics) for num_examples, metrics in results if metrics is not None]
    return aggregate_evaluation_metrics(successful_results)


def get_on_fit_config(save_path):
    """Return a function that will be used to construct the config that the client's
    fit() method will receive."""

    def fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["save_path"] = save_path
        return fit_config

    return fit_config_fn


# def fit_weighted_average(metrics):
#     """Aggregate (federated) evaluation metrics."""

#     if not metrics:
#         # Log the issue and return a default value to avoid the error
#         log(INFO, "'No metrics received for aggregation.")
#         return {}
    
#     log(INFO, f"Received metrics: {metrics}")
    
#     # aggregated_metrics = {
#     #     "eval_loss": 0,
#     #     "R1": 0,
#     #     "R2": 0,
#     #     "RL": 0,
#     #     "RLsum": 0
#     # }
    
#     # Multiply accuracy of each client by number of examples used
#     losses = [num_examples * m["eval_loss"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # total_examples = sum(num_examples for num_examples, _ in metrics)

#     # for num_examples, m in metrics:
#     #     weight = num_examples / total_examples
#     #     aggregated_metrics["eval_loss"] += weight * m["eval_loss"]
#     #     aggregated_metrics["R1"] += weight * m["R1"]
#     #     aggregated_metrics["R2"] += weight * m["R2"]
#     #     aggregated_metrics["RL"] += weight * m["RL"]
#     #     aggregated_metrics["RLsum"] += weight * m["RLsum"]

#     # Aggregate and return custom metric (weighted average)
#     log(INFO, {"Aggregated training loss across clients. eval_loss": sum(losses) / sum(examples)})
#     # log(INFO, f"Aggregated metrics across clients: {aggregated_metrics}")

#     return {"eval_loss": sum(losses) / sum(examples)} 
#     # return aggregated_metrics

def fit_weighted_average(metrics):
    """Aggregate (federated) evaluation metrics."""

    if not metrics:
        log(INFO, "No metrics received for aggregation.")
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    aggregated_metrics = {
        "train_loss": 0,
        "eval_loss": 0,
        "R1": 0,
        "R2": 0,
        "RL": 0,
        "RLsum": 0
    }

    for num_examples, m in metrics:
        weight = num_examples / total_examples
        for key in aggregated_metrics:
            aggregated_metrics[key] += weight * m[key]

    log(INFO, f"Aggregated metrics across clients: {aggregated_metrics}")
    return aggregated_metrics


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    current_date = date.today()
    
    log_folder_name = current_date.strftime("%Y-%m-%d")
    log_save_path = os.path.join(os.getcwd(), f"logs/{log_folder_name}")
    os.makedirs(log_save_path, exist_ok=True)
    log_file_name = f"{log_save_path}/{current_time.time().strftime('%H-%M')}_server_app.log"
    fl.common.logger.configure(identifier="myFlowertuneLLM", filename=log_file_name)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    log(INFO, {"config": cfg})

    # Get initial model weights
    init_model = get_model(cfg.model)
    init_model_parameters = get_parameters(init_model)
    init_model_parameters = ndarrays_to_parameters(init_model_parameters)

    strategy_name = cfg.strategy.name.lower()
    log(INFO, {"FL Strategy": strategy_name})

    if strategy_name in STRATEGIES:

        if strategy_name == 'fedavg':
            from flwr.server.strategy import FedAvg

            # Define strategy
            strategy = FedAvg(
                fraction_fit=cfg.strategy.fraction_fit,
                # fraction_evaluate=cfg.strategy.fraction_evaluate,
                on_fit_config_fn=get_on_fit_config(save_path),
                fit_metrics_aggregation_fn=fit_weighted_average,
                # evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                initial_parameters=init_model_parameters,
                # evaluate_fn=get_evaluate_fn(
                #     cfg.model,
                #     cfg.train.save_every_round,
                #     num_rounds,
                #     save_path,
                #     dataset_name=cfg.dataset.name,
                #     batch_size=cfg.eval.batch_size
                # ),
                # on_evaluate_config_fn=evaluate_config
            )
            
        elif strategy_name == 'fedprox':
            from flwr.server.strategy import FedProx

            # Define strategy using FedProx
            strategy = FedProx(
                fraction_fit=cfg.strategy.fraction_fit,
                on_fit_config_fn=get_on_fit_config(save_path),
                fit_metrics_aggregation_fn=fit_weighted_average,
                initial_parameters=init_model_parameters,
                proximal_mu=cfg.strategy.proximal_mu  # Set proximal term coefficient
            )

        elif strategy_name == 'fedavgm':
            from flwr.server.strategy import FedAvgM

            # Define strategy using FedAvgM
            strategy = FedAvgM(
                fraction_fit=cfg.strategy.fraction_fit,
                on_fit_config_fn=get_on_fit_config(save_path),
                fit_metrics_aggregation_fn=fit_weighted_average,
                initial_parameters=init_model_parameters,
                server_momentum=cfg.strategy.server_momentum # set server momentum
            )

        elif strategy_name == 'fedadagrad':
            from flwr.server.strategy import FedAdagrad

            # Define strategy using FedAdagrad
            strategy = FedAdagrad(
                fraction_fit=cfg.strategy.fraction_fit,
                on_fit_config_fn=get_on_fit_config(save_path),
                fit_metrics_aggregation_fn=fit_weighted_average,
                initial_parameters=init_model_parameters,
                eta=cfg.strategy.eta, # set global learning rate
                tau=cfg.strategy.tau # set regularisation parameter
            )

        elif strategy_name == 'fedadam':
            from flwr.server.strategy import FedAdam

            # Define strategy using FedAdam
            strategy = FedAdam(
                fraction_fit=cfg.strategy.fraction_fit,
                on_fit_config_fn=get_on_fit_config(save_path),
                fit_metrics_aggregation_fn=fit_weighted_average,
                initial_parameters=init_model_parameters,
                eta=cfg.strategy.eta, # set global learning rate
                tau=cfg.strategy.tau, # set regularisation parameter
                beta_1=cfg.strategy.server_momentum # set momentum
            ) 

        config = ServerConfig(num_rounds=num_rounds)
    else:
        raise(f"Invalid strategy: {strategy_name}. Ensure that the strategy is in the defined list of strategies ({STRATEGIES}).")

    return ServerAppComponents(strategy=strategy, config=config)

# Flower ServerApp
app = ServerApp(server_fn=server_fn)

