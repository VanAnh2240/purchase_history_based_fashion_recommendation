import os
import pandas as pd

from src.evaluation.metrics import calculate_metrics, print_metrics_table
from config import RESULT_DIR


class Evaluator:
    def __init__(self):
        self.all_results = []

    def evaluate(self, dataset, feature, model_name, preds, targets):
        metrics = calculate_metrics(preds, targets)
        metrics = {k: round(v, 4) for k, v in metrics.items()}

        print_metrics_table(metrics, title=f"{dataset} | {feature} | {model_name}")

        row = {
            "dataset": dataset,
            "feature": feature or "none",
            "model": model_name,
            **metrics
        }

        self.all_results.append(row)

        # Save per-run CSV
        feature_name = feature or "none"
        save_dir = os.path.join(RESULT_DIR, dataset, feature_name, model_name)
        os.makedirs(save_dir, exist_ok=True)

        csv_path = os.path.join(save_dir, f"{feature_name}_{model_name}.csv")

        pd.DataFrame([row]).to_csv(
            csv_path,
            index=False,
            float_format="%.4f"
        )

        print(f"[Save result] {csv_path}")

        return metrics

    def save_report(self, dataset):
        df = pd.DataFrame(self.all_results)

        if df.empty:
            print("[REPORT] No data to save")
            return None

        sort_col = "F1-score" if "F1-score" in df.columns else "f1"
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False)

        report_path = os.path.join(RESULT_DIR, dataset, "report.csv")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        df.to_csv(report_path, index=False, float_format="%.4f")

        print("\n====== REPORT CONTENT ======")
        print(df.to_markdown(index=False))

        print(f"\n[DONE] Report saved at: {report_path}")

        return report_path