import os

from custom_evaluate import eval_metrics
from custom_plot_froc import plot_overall_froc


def compute_offline_metrics(
    preds_dir=None,
    ground_truth_dir=None,
    save_dir=None,
    metrics_filename="metrics.json",
    plot_froc=True,
    plot_froc_single_wsis=False,
    froc_plot_filename="froc_curves_aggregated.png",
):
    assert preds_dir is not None, "Predictions directory is not set."
    assert ground_truth_dir is not None, "Ground truth directory is not set."
    assert save_dir is not None, "Save directory is not set."

    os.makedirs(save_dir, exist_ok=True)

    print("Computing metrics...")

    metrics = eval_metrics(
        predictions_folder=preds_dir,
        ground_truth_folder=ground_truth_dir,
        save_path=save_dir,
        filename=metrics_filename,
    )

    print(
        f"FROC scores for lymphocytes: {metrics['aggregates']['lymphocytes']['froc_score_aggr']}"
    )
    print(
        f"FROC scores for monocytes: {metrics['aggregates']['monocytes']['froc_score_aggr']}"
    )
    print(
        f"FROC scores for inflammatory-cells: {metrics['aggregates']['inflammatory-cells']['froc_score_aggr']}"
    )

    metrics_file_path = os.path.join(save_dir, metrics_filename)
    assert os.path.exists(metrics_file_path), "Metrics file not found!."

    if plot_froc is True:
        print("Plotting FROC curve(s)...")
        plot_overall_froc(
            input_path=metrics_file_path,
            output_path=save_dir,
            plot_per_file=plot_froc_single_wsis,
            filename=froc_plot_filename,
        )


if __name__ == "__main__":
    preds_dir = "/work/grana_urologia/MONKEY_challenge/outputs/detectron2_pretrained_True_e5000_b10_lr0.001/results/fold_4"
    ground_truth_dir = (
        "/work/grana_urologia/MONKEY_challenge/data/monkey-data/annotations/json_mm"
    )
    save_dir = "/work/grana_urologia/MONKEY_challenge/outputs/detectron2_pretrained_True_e5000_b10_lr0.001/results"
    metrics_filename = "metrics_fold_4.json"
    froc_plot_filename = "froc_curves_aggregated_fold_4.png"

    compute_offline_metrics(
        preds_dir=preds_dir,
        ground_truth_dir=ground_truth_dir,
        save_dir=save_dir,
        metrics_filename=metrics_filename,
        plot_froc=True,
        plot_froc_single_wsis=False,
        froc_plot_filename=froc_plot_filename,
    )
