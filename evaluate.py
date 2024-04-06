import os
import time
import sys
import argparse
import gc
import json
import logging
import os
from dataclasses import asdict

import torch

from evomerge import instantiate_from_config, load_config, set_seed

# logger = logging.getLogger(__name__)
# Configure logging to file and console
def setup_logging(output_path):
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"
    log_level = logging.INFO
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # File handler for outputting logs to a file
    file_handler = logging.FileHandler(output_path)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    # Console handler for outputting logs to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="config path")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    # validation
    if args.output_path is None:
        args.output_path = (
            os.path.splitext(os.path.basename(args.config_path))[0] + ".json"
        )
        args.output_path = f"results/{args.output_path}"
        os.makedirs("results", exist_ok=True)
    assert args.output_path.endswith(".json"), "`output_path` must be json file"
    return args


def main(args):
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )

    # Set up logging
    log_file_path = args.output_path.replace('.json', '.log') # Log file has the same name as output but with .log extension
    logger = setup_logging(log_file_path)

    config = load_config(args.config_path)
    logger.info(f"Config:\n{json.dumps(config, indent=2, ensure_ascii=False)}")
    set_seed(42)

    # 1. load model (it's already moved to device)
    model = instantiate_from_config(config["model"])
    logger.info(f"Model: {model.__class__.__name__}")

    eval_configs = config["eval"]
    if isinstance(eval_configs, dict):
        eval_configs = [eval_configs]

    results = {}
    for eval_config in eval_configs:
        # 2. load evaluator
        evaluator = instantiate_from_config(eval_config)
        logger.info(f"Evaluator: {evaluator.__class__.__name__}")
        # 3. Run!
        outputs = evaluator(model)
        logger.info(f"Result:\n{outputs.metrics}")
        results[evaluator.name] = asdict(outputs)
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    start_time = time.time()  # Capture start time
    args = parse_args()
    main(args)
    end_time = time.time()  # Capture end time

    # Calculate and log total runtime
    total_runtime_seconds = end_time - start_time  # Calculate total runtime
    total_runtime_minutes = total_runtime_seconds / 60  # Convert seconds to minutes
    logging.getLogger(__name__).info(f"Total runtime: {total_runtime_seconds:.2f} seconds / ({total_runtime_minutes:.2f} minutes)")
