"""Run walk-forward backtests on the LSTM predictor model for all tickers in eval_set.json and report performance."""

import json
import os
import sys

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(__file__))

from predictor import backtest_stock

def run_predictor_evaluation():
    eval_set_path = os.path.join(os.path.dirname(__file__), "eval_set.json")
    if not os.path.exists(eval_set_path):
        print(f"Error: {eval_set_path} not found.")
        return

    with open(eval_set_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    tickers = [case["ticker"] for case in cases]
    print(f"Starting LSTM model accuracy evaluation on {len(tickers)} tickers: {', '.join(tickers)}")
    print("-" * 80)

    results = []
    for ticker in tickers:
        print(f"Backtesting {ticker}...", end="", flush=True)
        res = backtest_stock(ticker, days_back=7)
        if "error" in res:
            print(f" ERROR: {res['error']}")
        else:
            print(f" Done (Accuracy: {res['direction_accuracy']}%, Error: {res['avg_price_error']}%, Grade: {res['grade']})")
            results.append(res)

    if not results:
        print("No successful backtests completed.")
        return

    # Print summary table
    headers = ["Ticker", "Cutoff Date", "Dir Accuracy (%)", "Avg Price Error (%)", "Pred Dir", "Act Dir", "Correct?", "Grade"]
    table_data = []
    
    total_dir_acc = 0.0
    total_price_err = 0.0
    correct_overall_dir_count = 0
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    for r in results:
        total_dir_acc += r["direction_accuracy"]
        total_price_err += r["avg_price_error"]
        if r["direction_correct"]:
            correct_overall_dir_count += 1
        grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1

        table_data.append([
            r["ticker"],
            r["cutoff_date"],
            f"{r['direction_accuracy']}%",
            f"{r['avg_price_error']}%",
            r["overall_direction"],
            r["actual_direction"],
            "Yes" if r["direction_correct"] else "No",
            r["grade"]
        ])

    print("\n" + "="*80)
    print("                      LSTM MODEL ACCURACY REPORT (7-Day Backtest)")
    print("="*80)
    
    # Simple manual markdown table formatting to avoid external dependency issues
    col_widths = [10, 15, 18, 20, 10, 10, 10, 8]
    
    def print_row(row):
        line = " | ".join(str(val).ljust(width) for val, width in zip(row, col_widths))
        print(f"| {line} |")

    # Print headers
    print_row(headers)
    print(f"|{'-' * (col_widths[0]+2)}|{'-' * (col_widths[1]+2)}|{'-' * (col_widths[2]+2)}|{'-' * (col_widths[3]+2)}|{'-' * (col_widths[4]+2)}|{'-' * (col_widths[5]+2)}|{'-' * (col_widths[6]+2)}|{'-' * (col_widths[7]+2)}|")
    
    for row in table_data:
        print_row(row)

    print("="*80)
    
    # Calculate averages
    n = len(results)
    avg_dir_acc = total_dir_acc / n
    avg_price_err = total_price_err / n
    overall_dir_acc_rate = (correct_overall_dir_count / n) * 100

    print("AGGREGATE METRICS:")
    print(f"  * Average Daily Directional Accuracy: {avg_dir_acc:.1f}%")
    print(f"  * Average Forecast Price Error:       {avg_price_err:.2f}%")
    print(f"  * Overall 7-Day Trend Accuracy Rate:  {overall_dir_acc_rate:.1f}% ({correct_overall_dir_count}/{n} tickers)")
    print("  * Distribution of Grades:")
    for g, count in sorted(grade_counts.items()):
        print(f"      Grade {g}: {count}")
    print("="*80)

    # Save to a json file
    output_path = os.path.join(os.path.dirname(__file__), "predictor_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "avg_daily_direction_accuracy": round(avg_dir_acc, 2),
                "avg_price_error": round(avg_price_err, 2),
                "overall_trend_accuracy_rate": round(overall_dir_acc_rate, 2),
                "grade_distribution": grade_counts
            },
            "results": results
        }, f, indent=2)
    print(f"Detailed evaluation results saved to: {output_path}")

if __name__ == "__main__":
    run_predictor_evaluation()
