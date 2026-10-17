import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import logging


logger = logging.getLogger(__name__)


def uq_results(df, save_location=None, prefix=None):
    # Plot standard deviations without modifying the caller's variance columns.
    df = df.copy()
    true_labels = df["true_label"].values
    probability_cols = sorted((c for c in df.columns if c.startswith("pred_conf") and c[9:].isdigit()), key=lambda c: int(c[9:]))
    pred_probs = df[probability_cols].values
    true_labels, pred_probs = _classification_inputs(true_labels, pred_probs)

    if "entropy" in df:
        evidential = df["entropy"]
        df_cols = ["aleatoric", "epistemic", "total", "entropy"]
        uq_names = ["Aleatoric", "Epistemic", "Total", "Entropy"]
    else:
        evidential = df["evidential"]
        df_cols = ["aleatoric", "epistemic", "total", "evidential"]
        uq_names = ["Aleatoric", "Epistemic", "Total", "Evidential"]

    df["total"] = df["aleatoric"] + df["epistemic"]
    df["epistemic"] = np.sqrt(df["epistemic"])
    df["aleatoric"] = np.sqrt(df["aleatoric"])
    df["total"] = np.sqrt(df["total"])

    # Compare uncertainties
    if "entropy" in df:
        plot_uncertainties(
            df,
            ["aleatoric", "total"],
            ["epistemic", "entropy"],
            x_labels=["Epistemic", "Entropy"],
            y_labels=["Aleatoric", "Total"],
            num_bins=20,
            save_location=save_location
        )
    else:
        plot_uncertainties(
            df,
            ["aleatoric", "total"],
            ["epistemic", "evidential"],
            x_labels=["Epistemic", "Evidential"],
            y_labels=["Aleatoric", "Total"],
            num_bins=20,
            save_location=save_location
        )

    # Create reliability diagrams for each class
    classifier_attribution(
        true_labels, pred_probs, save_location=save_location, prefix=prefix
    )

    # Create skill-score figures
    classifier_skill_scores(
        true_labels,
        pred_probs,
        df["aleatoric"],
        df["epistemic"],
        df["total"],
        evidential,
        num_bins=10,
        legend_cols=uq_names,
        save_location=save_location,
        prefix=prefix,
    )

    # Create discard fraction figures for each class
    classifier_discard_fraction(
        df,
        save_location=save_location,
        plt_titles=uq_names,
        uncertainty_cols=df_cols,
        num_classes=pred_probs.shape[1],
        prefix=prefix,
    )


def _classification_inputs(targets, probs):
    targets = np.asarray(targets)
    probs = np.asarray(probs, dtype=float)
    if targets.ndim != 1 or probs.ndim != 2 or len(targets) != len(probs) or len(targets) == 0 or probs.shape[1] < 2:
        raise ValueError("Provide one class label per row of a nonempty probability matrix")
    if not np.all(np.isfinite(probs)) or np.any((probs < 0) | (probs > 1)) or not np.allclose(probs.sum(axis=1), 1, rtol=1e-5, atol=1e-7):
        raise ValueError("Predictions must be finite, nonnegative class probabilities summing to one")
    if not np.issubdtype(targets.dtype, np.number) or not np.all(np.isfinite(targets)) or np.any(targets != np.floor(targets)) or np.any((targets < 0) | (targets >= probs.shape[1])):
        raise ValueError("Class labels must be integer indices into probability columns")
    return targets.astype(int), probs


def sort_arr(true_labels, pred_probs, confidences, n_bins=10, data_min=False, data_max=False):
    """Bin paired predictions, including constant values and the upper endpoint."""
    true_labels, pred_probs = _classification_inputs(true_labels, pred_probs)
    confidences = np.asarray(confidences, dtype=float)
    if confidences.shape != true_labels.shape or not np.all(np.isfinite(confidences)):
        raise ValueError("Provide one finite binning value per sample")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    data_min = confidences.min() if data_min is False or data_min is None else float(data_min)
    data_max = confidences.max() if data_max is False or data_max is None else float(data_max)
    if not np.isfinite([data_min, data_max]).all() or data_min > data_max:
        raise ValueError("Binning bounds must be finite and increasing")
    if data_min == data_max:
        delta = max(abs(data_min) * 1e-6, 1e-6)
        data_min, data_max = data_min - delta, data_max + delta
    if data_min > 0 and data_max - data_min > 10 * np.std(confidences):
        bins = np.geomspace(data_min, data_max, n_bins + 1)
    else:
        bins = np.linspace(data_min, data_max, n_bins + 1)
    centers = (bins[1:] + bins[:-1]) / 2
    pred_labels = pred_probs.argmax(axis=1)
    climatology = np.bincount(true_labels, minlength=pred_probs.shape[1]) / len(true_labels)
    results = defaultdict(list)
    for i in range(n_bins):
        selected = (confidences >= bins[i]) & ((confidences <= bins[i + 1]) if i == n_bins - 1 else (confidences < bins[i + 1]))
        if selected.any():
            results["bss"].append(brier_multi(true_labels[selected], pred_probs[selected], climatology=climatology))
            results["eb"].append(1 / np.sqrt(selected.sum()))
            results["acc"].append(np.mean(true_labels[selected] == pred_labels[selected]))
            results["count"].append(selected.sum())
            results["bin"].append(centers[i])
    return np.asarray(results["bin"]), results


def brier_multi(targets, probs, num_classes=None, skill_score=True, climatology=None):
    """Class-mean Brier score, or skill against per-class climatological frequencies.

    The class normalization is identical in forecast and reference scores.
    Skill is undefined (NaN) when the reference score is zero.
    """
    targets, probs = _classification_inputs(targets, probs)
    if num_classes is not None and num_classes != probs.shape[1]:
        raise ValueError("num_classes must match probability columns")
    one_hot = np.eye(probs.shape[1])[targets]
    score = np.mean((probs - one_hot) ** 2)
    if not skill_score:
        return score
    reference = one_hot.mean(axis=0) if climatology is None else np.asarray(climatology, dtype=float)
    if reference.shape != (probs.shape[1],) or np.any(reference < 0) or not np.isfinite(reference).all() or not np.isclose(reference.sum(), 1):
        raise ValueError("climatology must contain one probability per class")
    baseline = np.mean((one_hot - reference) ** 2)
    return 1 - score / baseline if baseline > 0 else np.nan


def brier_skill_score_multiclass(true_labels, pred_probs, skill_score=True):
    """Mean of one-vs-rest class Brier scores (or individual class skills).

    For skill, classes absent from or universal in the observations have an
    undefined climatological skill; the macro skill is therefore NaN.
    """
    true_labels, pred_probs = _classification_inputs(true_labels, pred_probs)
    observations = np.eye(pred_probs.shape[1])[true_labels]
    scores = np.mean((observations - pred_probs) ** 2, axis=0)
    if not skill_score:
        return scores.mean()
    frequencies = observations.mean(axis=0)
    reference = frequencies * (1 - frequencies)
    if np.any(reference == 0):
        return np.nan
    return np.mean(1 - scores / reference)


def plot_uncertainties(
    df,
    input_cols,
    output_cols,
    num_bins=20,
    legend_cols=None,
    x_labels=None,
    y_labels=None,
    fontsize=10,
    save_location=None,
):

    width = 5 if len(output_cols) == 1 else 10
    height = 3.5 if len(output_cols) == 1 else 3.5
    fig, axs = plt.subplots(1, len(output_cols), figsize=(width, height))
    if len(output_cols) == 1:
        axs = [axs]

    if legend_cols is None:
        legend_cols = output_cols

    if len(input_cols) != len(output_cols) or not output_cols:
        raise ValueError("Provide matching nonempty input and output columns")
    x_labels = output_cols if x_labels is None else x_labels
    y_labels = input_cols if y_labels is None else y_labels
    not_nan = np.isfinite(df[input_cols + output_cols]).all(axis=1)
    df = df.loc[not_nan].copy()
    if df.empty:
        raise ValueError("At least one finite paired sample is required")
    
    # Loop over each element in output_cols and create a hexbin plot
    for i, (i_col, o_col) in enumerate(zip(input_cols, output_cols)):
        # Calculate the mean prediction for the current column
        y_array = df[i_col].copy()
        x_array = df[o_col].copy()

        # Create the 2D histogram plot
        my_range = [
            [np.nanpercentile(x_array, 5), np.nanpercentile(x_array, 95)],
            [np.nanpercentile(y_array, 5), np.nanpercentile(y_array, 95)],
        ]
        inside = ((x_array >= my_range[0][0]) & (x_array <= my_range[0][1]) &
                  (y_array >= my_range[1][0]) & (y_array <= my_range[1][1]))
        if not inside.any():
            my_range = [[x_array.min(), x_array.max()], [y_array.min(), y_array.max()]]
        hb = axs[i].hist2d(
            x_array,
            y_array,
            bins=num_bins,
            cmap="inferno",
            range=my_range,
            norm=colors.LogNorm(vmin=1, vmax=max(2, len(df))),
        )

        # Set the axis labels
        # axs[i].set_title(legend_cols[i], fontsize=fontsize)
        axs[i].set_xlabel(x_labels[i], fontsize=fontsize)

        axs[i].set_ylabel(y_labels[i], fontsize=fontsize)
        axs[i].tick_params(axis="both", which="major", labelsize=fontsize)

        # Move the colorbar below the x-axis label
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("bottom", size="5%", pad=0.6)
        cbar = fig.colorbar(hb[3], cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)

        # Set the tick labels to use scientific notation
        axs[i].ticklabel_format(style="sci", axis="both", scilimits=(-1, 1))

        # Add 1-1 line
        smaller = min(my_range[0][0], my_range[1][0])
        larger = max(my_range[0][1], my_range[1][1])
        axs[i].plot(
            np.linspace(smaller, larger, 10),
            np.linspace(smaller, larger, 10),
            ls="--",
            c="k",
        )

    # make it pretty
    plt.tight_layout()

    if save_location:
        plt.savefig(
            os.path.join(save_location, "compare_uncertanties.pdf"),
            dpi=300,
            bbox_inches="tight",
        )


def classifier_attribution(
    true_labels,
    pred_probs,
    num_bins=10,
    legend_cols=["Rain", "Snow", "Sleet", "Frz Rain"],
    save_location=False,
    prefix=False,
):

    true_labels, pred_probs = _classification_inputs(true_labels, pred_probs)
    num_classes = pred_probs.shape[1]
    if legend_cols is None or len(legend_cols) != num_classes:
        legend_cols = [str(i) for i in range(num_classes)]
    fig, axs = plt.subplots(2, num_classes, figsize=(10, 5), sharey="row", sharex="col", squeeze=False)

    for true_label in range(num_classes):
        # Top row figures
        bin_centers, results = _class_reliability(true_labels == true_label, pred_probs[:, true_label], 50)
        _ = axs[0][true_label].hist2d(
            bin_centers,
            results["acc"],
            weights=results["count"],
            bins=num_bins,
            range=((0, 1), (0, 1)),
            cmap="inferno",
        )
        axs[0][true_label].plot(
            np.linspace(0, 1, 10), np.linspace(0, 1, 10), "b:", lw=3
        )
        axs[0][true_label].set_title(legend_cols[true_label], fontsize=10)

        # Bottom row
        bin_centers, results = _class_reliability(true_labels == true_label, pred_probs[:, true_label], num_bins)
        axs[1][true_label].errorbar(
            bin_centers, results["acc"], yerr=results["eb"], color="r"
        )

        bin_centers = np.linspace(0, 1, 100)
        ave_true = np.mean(true_labels == true_label)
        ave_true_range = np.array([ave_true for x in bin_centers])

        no_skill = [0.5 * x1 + 0.5 * ave_true for x1 in bin_centers]
        axs[1][true_label].plot(bin_centers, no_skill, "b-")
        axs[1][true_label].plot(
            np.linspace(0, 1, len(ave_true_range)), ave_true_range, ls=":", color="k"
        )
        axs[1][true_label].plot(
            ave_true_range, np.linspace(0, 1, len(ave_true_range)), ls=":", color="k"
        )
        axs[1][true_label].plot(bin_centers, bin_centers, ls=":", color="k")

        fill_cond = np.where(bin_centers < ave_true)[0]
        axs[1][true_label].fill_between(
            np.array(bin_centers)[fill_cond],
            0,
            np.array(no_skill)[fill_cond],
            color="lightblue",
        )
        fill_cond = np.where(bin_centers >= ave_true)[0]
        axs[1][true_label].fill_between(
            np.array(bin_centers)[fill_cond],
            1,
            np.array(no_skill)[fill_cond],
            color="lightblue",
        )
        axs[1][true_label].set_xlim([0, 1.01])
        axs[1][true_label].set_ylim([0, 1.01])
        # axs[1][true_label].set_title(legend_cols[true_label], fontsize = 10)
        axs[1][true_label].set_xlabel("Predicted Probability")

    axs[0][0].set_ylabel("Conditional Observed\n Probability")
    axs[1][0].set_ylabel("Conditional Observed\n Probability")

    if save_location:
        if prefix:
            name = f"class_attr_{prefix}.pdf"
        else:
            name = "class_attr.pdf"
        plt.savefig(
            os.path.join(save_location, name),
            dpi=300,
            bbox_inches="tight",
        )


def classifier_skill_scores(
    true_labels,
    pred_probs,
    aleatoric,
    epistemic,
    total,
    evidential,
    num_bins=10,
    legend_cols=["Aleatoric", "Epistemic", "Total", "Evidential"],
    save_location=False,
    prefix=False,
):

    fig, axs = plt.subplots(1, 4, figsize=(10, 3.5), sharey="row")

    for k, uq in enumerate([aleatoric, epistemic, total, evidential]):
        bin_centers, results = sort_arr(
            true_labels, pred_probs, uq, num_bins * num_bins
        )
        rescaled_bin_centers = ((bin_centers - np.min(bin_centers)) / np.ptp(bin_centers)
                                if np.ptp(bin_centers) > 0 else np.zeros_like(bin_centers))
        _ = axs[k].hist2d(
            rescaled_bin_centers,
            results["bss"],
            weights=results["count"],
            bins=(num_bins, num_bins),
            range=((0, 1), (-0.25, 1)),
            cmap="inferno",
        )
        axs[k].plot(np.linspace(0, 1, 10), np.linspace(1, 0, 10), "b:", lw=3)
        axs[k].set_title(legend_cols[k], fontsize=10)
        axs[k].set_xlabel("Rescaled uncertainty", fontsize=10)
        # axs[k].set_xscale("log")

        ax2 = axs[k].twiny()
        lower, upper = min(uq), max(uq)
        padding = max(abs(lower) * 1e-6, 1e-6) if lower == upper else 0
        ax2.set_xlim(lower - padding, upper + padding)
        # ax2.set_xticklabels = bin_centers

    axs[0].set_ylabel("Brier Skill Score", fontsize=10)

    plt.tight_layout()
    if save_location:
        if prefix:
            name = f"class_skill_{prefix}.pdf"
        else:
            name = "class_skill.pdf"
        plt.savefig(
            os.path.join(save_location, name),
            dpi=300,
            bbox_inches="tight",
        )


def classifier_discard_fraction(
    df,
    num_bins=10,
    num_classes=4,
    uncertainty_cols=["aleatoric", "epistemic", "total", "evidential"],
    legend_cols=["Rain", "Snow", "Sleet", "Frz Rain"],
    plt_titles=["Aleatoric", "Epistemic", "Total", "Evidential"],
    colors=["#f8d605", "#ce4912", "#042c71", "b", "g"],
    save_location=False,
    prefix=False,
):

    if not uncertainty_cols:
        raise ValueError("At least one uncertainty column is required")
    fig, axs = plt.subplots(1, len(uncertainty_cols), figsize=(10, 3.5), sharey="row", squeeze=False)
    axs = axs[0]
    if len(legend_cols) != num_classes:
        legend_cols = [str(index) for index in range(num_classes)]
    if len(plt_titles) != len(uncertainty_cols):
        plt_titles = uncertainty_cols
    probability_cols = [f"pred_conf{k+1}" for k in range(num_classes)]
    labels, probabilities = _classification_inputs(df["true_label"].to_numpy(), df[probability_cols].to_numpy())
    climatology = np.bincount(labels, minlength=num_classes) / len(labels)
    db = (1.0 / num_classes) * (100 / num_bins)
    # df["total"] = np.sqrt(df["aleatoric"] + df["epistemic"])

    for k, uq in enumerate(uncertainty_cols):

        dg = df.copy().sort_values(uq)

        for class_label in range(num_classes):
            dr = dg[dg["true_label"] == class_label].copy()
            results = defaultdict(list)
            percentage = np.linspace(0, 1, dr.shape[0])
            for percent in np.linspace(100 / num_bins, 100, num_bins):
                c = percentage <= percent / 100.0
                if c.sum() == 0:
                    continue
                acc = (dr[c]["true_label"] == dr[c]["pred_label"]).mean()
                # bss = dr[c]["bss"].mean()
                bss = brier_multi(
                    dr[c]["true_label"].values,
                    dr[c][[f"pred_conf{k+1}" for k in range(num_classes)]].values,
                    climatology=climatology,
                )
                results["acc"].append(acc)
                results["bss"].append(bss)
                results["frac"].append(percent)

            axs[k].bar(
                [100 - x + class_label * db for x in results["frac"]],
                results["bss"],
                db,
                color=colors[class_label % len(colors)],
            )

        axs[k].set_xlabel("Fraction removed")
        axs[k].set_title(plt_titles[k])
        axs[k].legend(legend_cols, loc="best")
        # axs[k].set_yscale("log")

    axs[0].set_ylabel("Brier Skill Score")

    plt.tight_layout()

    if save_location:
        if prefix:
            name = f"discard_fraction_{prefix}.pdf"
        else:
            name = "discard_fraction.pdf"
        plt.savefig(
            os.path.join(save_location, name),
            dpi=300,
            bbox_inches="tight",
        )


def _class_reliability(observed, probabilities, num_bins):
    """One-vs-rest observed frequency conditional on the forecast probability."""
    bins = np.linspace(0, 1, num_bins + 1)
    assigned = np.minimum(np.searchsorted(bins, probabilities, side="right") - 1, num_bins - 1)
    results = defaultdict(list)
    for index in range(num_bins):
        selected = assigned == index
        if selected.any():
            frequency = np.mean(observed[selected])
            results["bin"].append(np.mean(probabilities[selected]))
            results["acc"].append(frequency)
            # Preserve the original 1/sqrt(count) display-error convention.
            results["eb"].append(1 / np.sqrt(selected.sum()))
            results["count"].append(selected.sum())
    return np.asarray(results["bin"]), results
