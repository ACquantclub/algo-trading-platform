import json
import os
import importlib.util
import pandas as pd
from .portfolio_evaluator import PortfolioEvaluator
from datetime import datetime


class Grader:
    def __init__(self):
        self.setup_market_data()

    def setup_market_data(self):
        """Load and prepare market data"""
        with open("/autograder/source/data/testing.json", "r") as f:
            price_data = pd.read_json(f)

        market_data = {}
        for field in ["open", "close", "high", "low", "volume"]:
            data = price_data[field].to_dict()
            df = pd.DataFrame.from_dict(data, orient="index")
            df.index = pd.to_datetime(df.index)
            market_data[field] = df

        returns = market_data["close"].pct_change().dropna()
        self.evaluator = PortfolioEvaluator(market_data, returns)

    def find_strategy_file(self):
        """Find the first Python file in submission directory containing a class with allocate method."""
        submission_dir = "/autograder/submission"
        for file in os.listdir(submission_dir):
            if file.endswith(".py"):
                file_path = os.path.join(submission_dir, file)
                try:
                    spec = importlib.util.spec_from_file_location(
                        "strategy_module", file_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if (
                            isinstance(item, type)
                            and hasattr(item, "allocate")
                            and callable(getattr(item, "allocate"))
                        ):
                            return item
                except Exception:
                    continue
        raise FileNotFoundError("No valid strategy file found in submission directory")

    def grade_submission(self):
        """Grade the submission and return results in Gradescope format"""
        results = {
            "score": 0,
            "execution_time": 0,
            "output": "",
            "visibility": "visible",
            "stdout_visibility": "visible",
            "tests": [],
            "leaderboard": [],
        }

        start_time = datetime.now()

        try:
            # Test submission format
            strategy_class = self.find_strategy_file()
            strategy_instance = strategy_class()

            results["tests"].append(
                {
                    "score": 1,
                    "max_score": 1,
                    "name": "Submission Format Check",
                    "output": "Valid strategy class found with allocate method",
                    "status": "passed",
                    "visibility": "visible",
                }
            )
            results["score"] += 1

            # Evaluate strategy performance
            eval_results = self.evaluator.evaluate_strategy(strategy_class)

            if not eval_results:
                raise RuntimeError("No results returned from evaluate_strategy")

            # Create leaderboard entries
            metrics = {
                "sharpe_ratio": eval_results["sharpe_ratio"],
                "sortino_ratio": eval_results["sortino_ratio"],
                "annual_return": eval_results["annual_return"],
                "max_drawdown": eval_results["max_drawdown"],
                "volatility": eval_results["volatility"],
                "win_rate": eval_results["win_rate"],
                "calmar_ratio": eval_results["calmar_ratio"],
            }

            # Add metrics to leaderboard
            for metric_name, value in metrics.items():
                results["leaderboard"].append(
                    {
                        "name": metric_name,
                        "value": round(float(value), 3),
                        "order": "asc" if metric_name == "max_drawdown" else "desc",
                    }
                )

            results["tests"].append(
                {
                    "score": 1,
                    "max_score": 1,
                    "name": "Strategy Evaluation",
                    "output": f"Strategy evaluated successfully. \nMetrics: {json.dumps(metrics, indent=2)}",
                    "status": "passed",
                    "visibility": "visible",
                }
            )
            results["score"] += 1

        except Exception as e:
            results["output"] = f"Error during grading: {str(e)}"
            results["tests"].append(
                {
                    "score": 0,
                    "max_score": 2,
                    "name": "Strategy Evaluation",
                    "output": f"Error: {str(e)}",
                    "status": "failed",
                    "visibility": "visible",
                }
            )

        results["execution_time"] = (datetime.now() - start_time).total_seconds()
        return results
