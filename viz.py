import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def save_to_csv(df, out_file="results_summary.csv"):
    df.to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")

def load_results(base_dir="results"):
    rows = []
    for exp in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp)
        config_path = os.path.join(exp_path, "config.json")
        summary_path = os.path.join(exp_path, "summary_eval_result.json")

        if not (os.path.isfile(config_path) and os.path.isfile(summary_path)):
            continue

        with open(config_path, "r") as f:
            config = json.load(f)
        with open(summary_path, "r") as f:
            summary = json.load(f)

        noise_args = config["Noise Arguments"]
        noise_method = config["Noise Method"] if noise_args["p"] > 0 else "no_noise"

        row = {
            "Experiment": exp,
            "Data Loader": config["Data Loader"],
            "NMF Method": config["NMF Method"],
            "Noise Method": noise_method,
        }

        for k, v in noise_args.items():
            row[k] = v
        for metric, stats in summary.items():
            row[f"{metric}_mu"] = stats["mu"]
            row[f"{metric}_sd"] = stats["sd"]

        rows.append(row)
    df = pd.DataFrame(rows)
    no_noise_df = pd.DataFrame()
    for (data_loader, nmf_method), group_df in df.groupby(
        ["Data Loader", "NMF Method"]
    ):
        group_no_noise_df = group_df[group_df["Noise Method"] == "no_noise"]
        no_noise_df = pd.concat(
            [
                no_noise_df,
                pd.DataFrame(
                    {
                        "Data Loader": [data_loader],
                        "NMF Method": [nmf_method],
                        **{
                            f"no_noise_{col.replace('_mu', '')}": group_no_noise_df[col]
                            for col in group_no_noise_df.columns
                            if col.endswith("_mu")
                        },
                    }
                ),
            ]
        )
    df = df.merge(no_noise_df)
    metrics = [col.replace("_mu", "") for col in df.columns if col.endswith("_mu")]
    for metric in metrics:
        df[f"perc_diff_to_no_noise_{metric}"] = (
            df[f"no_noise_{metric}"] - df[f"{metric}_mu"]
        ) / df[f"no_noise_{metric}"]

    return df


def plot_salt_pepper_grid(df, out_file="viz/salt_pepper_grid_5.png"):
    df_sp = df[
        (df["Noise Method"] == "salt_and_pepper")
        & (df["Data Loader"] == "DataLoaderORL")
    ]

    if df_sp.empty:
        print("No salt-and-pepper results found!")
        return

    p_vals = sorted(df_sp["p"].unique())
    r_vals = sorted(df_sp["r"].unique())

    fig, axes = plt.subplots(
        len(r_vals),
        len(p_vals),
        figsize=(4 * len(p_vals), 3 * len(r_vals)),
        sharey=True,
    )

    if len(r_vals) == 1 and len(p_vals) == 1:
        axes = [[axes]]  # special case
    elif len(r_vals) == 1:
        axes = [axes]  # single row
    elif len(p_vals) == 1:
        axes = [[ax] for ax in axes]  # single column

    colors = {"ACC_mu": "tab:blue", "RRE_mu": "tab:orange", "NMI_mu": "tab:green"}

    for i, r in enumerate(r_vals):
        for j, p in enumerate(p_vals):
            ax = axes[i][j]
            subset = df_sp[(df_sp["p"] == p) & (df_sp["r"] == r)]

            if subset.empty:
                ax.axis("off")
                continue

            # Melt into long form for plotting
            plot_df = subset.melt(
                id_vars=["NMF Method"],
                value_vars=["ACC_mu", "RRE_mu", "NMI_mu"],
                var_name="Metric",
                value_name="Value",
            )

            # sns.barplot(
            #     data=plot_df,
            #     hue="NMF Method",
            #     y="Value",
            #     x="Metric",
            #     palette="tab10",
            #     ax=ax,
            #     errorbar=None
            # )

            sns.pointplot(
                data=plot_df,
                hue="NMF Method",
                y="Value",
                x="Metric",
                palette="tab10",
                ax=ax,
                dodge=0.4,
                markers=["o", "s", "D"],  # circle, square, diamond
                linestyles="",
                errorbar=None,
            )

            # ax.set_title(f"p={p}, r={r}")
            # ax.set_xticks(range(len(subset["Metric"].unique())))
            # ax.set_xticklabels(subset["Metric"].unique(), rotation=30, ha="right")
            # ax.set_ylim(0, 1.0)
            # if j == 0:
            #     ax.set_ylabel("Score")
            # else:
            #     ax.set_ylabel("")
            # if i == len(r_vals)-1:
            #     ax.set_xlabel("Model")
            # else:
            #     ax.set_xlabel("")

            # ax.set_title(f"p={p}, r={r}")
            # ax.set_ylim(0, 1.0)  # assuming metrics normalized
            # if j == 0:
            #     ax.set_ylabel("Score")
            # else:
            #     ax.set_ylabel("")
            # if i == len(r_vals)-1:
            #     ax.set_xlabel("Model")
            # else:
            #     ax.set_xlabel("")

            ax.set_title(f"p={p}, r={r}")
            ax.set_ylim(0, 1.0)

            # Only label y-axis on the first column
            if j == 0:
                ax.set_ylabel("Score")
            else:
                ax.set_ylabel("")

            # Only label x-axis on the bottom row
            if i == len(r_vals) - 1:
                ax.set_xlabel("Metric")
            else:
                ax.set_xlabel("")

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    print(f"Saved figure to {out_file}")

    plt.show()


def plot_salt_and_pepper_noise_2(df, dataset, out_file="viz/salt_and_pepper_noise_2"):
    if dataset not in ["DataLoaderYale", "DataLoaderORL"]:
        raise ValueError("dataset must be 'DataLoaderYale' or 'DataLoaderORL'")
    
    folder = os.path.dirname(out_file)
    base_name = os.path.basename(out_file)

    if dataset == "DataLoaderYale":
        out_file = os.path.join(folder, f"yale_{base_name}")
    elif dataset == "DataLoaderORL":
        out_file = os.path.join(folder, f"orl_{base_name}")

    for r in sorted(df["r"].dropna().unique()):
        print(r)
        df_g = df[
            (df["Noise Method"] == "salt_and_pepper")
            & (df["Data Loader"] == dataset)
            & (df["r"] == r)
        ]
        if df_g.empty:
            print("No Salt and Pepper noise results found!")
            continue

        # models = df_g["NMF Method"].unique()
        metrics = ["RRE", "ACC", "NMI"]
        fig, axes = plt.subplots(
            len(metrics), 1, figsize=(8, 4 * len(metrics)), sharex=True
        )
        if dataset == "DataLoaderYale":
            fig.suptitle(
                f"Yale Dataset - Salt and Pepper Noise ($r={r}$)",
                fontsize=16,
            )
        elif dataset == "DataLoaderORL":
            fig.suptitle(
                f"ORL Dataset - Salt and Pepper Noise ($r={r}$)",
                fontsize=16,
            )

        if len(metrics) == 1:
            axes = [axes]

        colors = {
            "Lecture": "tab:blue",
            "KLDivergence": "tab:orange",
            "L21": "tab:green",
        }
        import numpy as np

        for i, metric in enumerate(metrics):
            plot_df = df_g[["NMF Method", "p", f"{metric}_mu", f"{metric}_sd"]]
            plot_df["upper"] = plot_df[f"{metric}_mu"] + plot_df[f"{metric}_sd"]
            plot_df["lower"] = plot_df[f"{metric}_mu"] - plot_df[f"{metric}_sd"]
            sns.lineplot(
                data=plot_df,
                x="p",
                y=f"{metric}_mu",
                hue="NMF Method",
                marker="o",
                palette=colors,
                ax=axes[i],
            )
            sns.lineplot(
                data=plot_df,
                x="p",
                y=f"upper",
                hue="NMF Method",
                marker="",
                alpha=0.2,
                palette=colors,
                ax=axes[i],
                legend=False,
            )
            sns.lineplot(
                data=plot_df,
                x="p",
                y=f"lower",
                hue="NMF Method",
                marker="",
                alpha=0.2,
                palette=colors,
                ax=axes[i],
                legend=False,
            )
            axes[i].set_title(f"Metric: {metric}")
            # axes[i].set_ylim(
            #     min(plot_df[f"{metric}_mu"]) * 0.9, min(plot_df[f"{metric}_mu"]) * 1.1
            # )
            axes[i].set_ylabel("Score")
            axes[i].legend(title="Model", loc="best")

        axes[-1].set_xlabel("Noise $p$")
        plt.tight_layout()
        os.makedirs(os.path.dirname(f"{out_file}_r_{r}.png"), exist_ok=True)
        plt.savefig(f"{out_file}_r_{r}.png", dpi=300)
        plt.show()
        print(f"Saved figure to {out_file}")


def plot_gaussian_noise(df, out_file="viz/gaussian_noise_1.png"):
    df_g = df[
        (df["Noise Method"] == "gaussian_noise")
        & (df["Data Loader"] == "DataLoaderORL")
    ]
    if df_g.empty:
        print("No Gaussian noise results found!")
        return

    models = df_g["NMF Method"].unique()
    fig, axes = plt.subplots(
        len(models), 1, figsize=(6, 4 * len(models)), sharex=True, sharey=True
    )
    fig.suptitle("Gaussian Noise: Metrics vs $\sigma$ for Each Model", fontsize=16)

    if len(models) == 1:
        axes = [axes]

    colors = {"ACC_mu": "tab:blue", "RRE_mu": "tab:orange", "NMI_mu": "tab:green"}

    for ax, model in zip(axes, models):
        subset = df_g[df_g["NMF Method"] == model]
        plot_df = subset.melt(
            id_vars=["sd"],
            value_vars=["ACC_mu", "RRE_mu", "NMI_mu"],
            var_name="Metric",
            value_name="Value",
        )
        sns.lineplot(
            data=plot_df,
            x="sd",
            y="Value",
            hue="Metric",
            marker="o",
            palette=colors,
            ax=ax,
        )
        ax.set_title(f"Model: {model}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.legend(title="Metric", loc="upper right")

    axes[-1].set_xlabel("Noise $\sigma$")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved figure to {out_file}")


def plot_uniform(df, out_file="viz/uniform_noise.png"):
    df_u = df[
        (df["Noise Method"] == "uniform_random")
        & (df["Data Loader"] == "DataLoaderORL")
    ]
    if df_u.empty:
        print("No Uniform noise results found!")
        return

    models = df_u["NMF Method"].unique()
    fig, axes = plt.subplots(
        len(models), 1, figsize=(6, 4 * len(models)), sharex=True, sharey=True
    )
    fig.suptitle("Uniform Noise: Metrics vs Variance for Each Model", fontsize=16)

    if len(models) == 1:
        axes = [axes]

    colors = {"ACC_mu": "tab:blue", "RRE_mu": "tab:orange", "NMI_mu": "tab:green"}

    for ax, model in zip(axes, models):
        subset = df_u[df_u["NMF Method"] == model]
        plot_df = subset.melt(
            id_vars=["var"],
            value_vars=["ACC_mu", "RRE_mu", "NMI_mu"],
            var_name="Metric",
            value_name="Value",
        )
        sns.lineplot(
            data=plot_df,
            x="var",
            y="Value",
            hue="Metric",
            marker="o",
            palette=colors,
            ax=ax,
        )
        ax.set_title(f"Model: {model}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.legend(title="Metric", loc="upper right")

    axes[-1].set_xlabel("Noise Variance")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved figure to {out_file}")


def plot_gaussian_noise_2(df, dataset, out_file="viz/gaussian_noise_2"):
    if dataset not in ["DataLoaderYale", "DataLoaderORL"]:
        raise ValueError("dataset must be 'DataLoaderYale' or 'DataLoaderORL'")
    
    folder = os.path.dirname(out_file)
    base_name = os.path.basename(out_file)

    if dataset == "DataLoaderYale":
        out_file = os.path.join(folder, f"yale_{base_name}.png")
    elif dataset == "DataLoaderORL":
        out_file = os.path.join(folder, f"orl_{base_name}.png")

    df_g = df[
        (df["Noise Method"] == "gaussian_noise")
        & (df["Data Loader"] == dataset)
    ]
    if df_g.empty:
        print("No Gaussian noise results found!")
        return

    # models = df_g["NMF Method"].unique()
    metrics = ["RRE", "ACC", "NMI"]
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(8, 4 * len(metrics)), sharex=True
    )
    if dataset == "DataLoaderYale":
        fig.suptitle("Yale Dataset - Gaussian Noise: Metrics vs $\sigma$ for Each Model", fontsize=16)
    elif dataset == "DataLoaderORL":
        fig.suptitle("ORL Dataset - Gaussian Noise: Metrics vs $\sigma$ for Each Model", fontsize=16)

    

    if len(metrics) == 1:
        axes = [axes]

    colors = {
        "Lecture": "tab:blue",
        "KLDivergence": "tab:orange",
        "L21": "tab:green",
    }
    import numpy as np

    for i, metric in enumerate(metrics):
        plot_df = df_g[["NMF Method", "sd", f"{metric}_mu", f"{metric}_sd"]]
        plot_df["upper"] = plot_df[f"{metric}_mu"] + plot_df[f"{metric}_sd"]
        plot_df["lower"] = plot_df[f"{metric}_mu"] - plot_df[f"{metric}_sd"]
        sns.lineplot(
            data=plot_df,
            x="sd",
            y=f"{metric}_mu",
            hue="NMF Method",
            marker="o",
            palette=colors,
            ax=axes[i],
        )
        sns.lineplot(
            data=plot_df,
            x="sd",
            y=f"upper",
            hue="NMF Method",
            marker="",
            alpha=0.2,
            palette=colors,
            ax=axes[i],
            legend=False,
        )
        sns.lineplot(
            data=plot_df,
            x="sd",
            y=f"lower",
            hue="NMF Method",
            marker="",
            alpha=0.2,
            palette=colors,
            ax=axes[i],
            legend=False,
        )
        axes[i].set_title(f"Metric: {metric}")
        # axes[i].set_ylim(
        #     min(plot_df[f"{metric}_mu"]) * 0.9, min(plot_df[f"{metric}_mu"]) * 1.1
        # )
        axes[i].set_ylabel("Score")
        axes[i].legend(title="Model", loc="best")

    axes[-1].set_xlabel("Noise $\sigma$")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved figure to {out_file}")


def plot_uniform_noise_2(df, dataset, out_file="viz/uniform_noise_2"):
    if dataset not in ["DataLoaderYale", "DataLoaderORL"]:
        raise ValueError("dataset must be 'DataLoaderYale' or 'DataLoaderORL'")
    
    folder = os.path.dirname(out_file)
    base_name = os.path.basename(out_file)

    if dataset == "DataLoaderYale":
        out_file = os.path.join(folder, f"yale_{base_name}.png")
    elif dataset == "DataLoaderORL":
        out_file = os.path.join(folder, f"orl_{base_name}.png")
        
    df_g = df[
        (df["Noise Method"] == "uniform_random")
        & (df["Data Loader"] == dataset)
    ]
    if df_g.empty:
        print("No uniform noise results found!")
        return

    # models = df_g["NMF Method"].unique()
    metrics = ["RRE", "ACC", "NMI"]
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(8, 4 * len(metrics)), sharex=True
    )
    if dataset == "DataLoaderYale":
        fig.suptitle("Yale Dataset - Uniform-Random Noise: Metrics vs $\phi$ for Each Model", fontsize=16)
    elif dataset == "DataLoaderORL":
        fig.suptitle("ORL Dataset - Uniform-Random Noise: Metrics vs $\phi$ for Each Model", fontsize=16)


    if len(metrics) == 1:
        axes = [axes]

    colors = {
        "Lecture": "tab:blue",
        "KLDivergence": "tab:orange",
        "L21": "tab:green",
    }
    import numpy as np

    for i, metric in enumerate(metrics):
        plot_df = df_g[["NMF Method", "var", f"{metric}_mu", f"{metric}_sd"]]
        plot_df["upper"] = plot_df[f"{metric}_mu"] + plot_df[f"{metric}_sd"]
        plot_df["lower"] = plot_df[f"{metric}_mu"] - plot_df[f"{metric}_sd"]
        sns.lineplot(
            data=plot_df,
            x="var",
            y=f"{metric}_mu",
            hue="NMF Method",
            marker="o",
            palette=colors,
            ax=axes[i],
        )
        sns.lineplot(
            data=plot_df,
            x="var",
            y=f"upper",
            hue="NMF Method",
            marker="",
            alpha=0.2,
            palette=colors,
            ax=axes[i],
            legend=False,
        )
        sns.lineplot(
            data=plot_df,
            x="var",
            y=f"lower",
            hue="NMF Method",
            marker="",
            alpha=0.2,
            palette=colors,
            ax=axes[i],
            legend=False,
        )
        axes[i].set_title(f"Metric: {metric}")
        # axes[i].set_ylim(
        #     min(plot_df[f"{metric}_mu"]) * 0.9, min(plot_df[f"{metric}_mu"]) * 1.1
        # )
        axes[i].set_ylabel("Score")
        axes[i].legend(title="Model", loc="best")

    axes[-1].set_xlabel("Noise $\phi$")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved figure to {out_file}")


def plot_gaussian_noise_3(df, out_file="viz/gaussian_noise_3.png"):
    df_g = df[
        (df["Noise Method"] == "gaussian_noise")
        & (df["Data Loader"] == "DataLoaderORL")
    ]
    if df_g.empty:
        print("No Gaussian noise results found!")
        return

    # models = df_g["NMF Method"].unique()
    metrics = ["RRE", "ACC", "NMI"]
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(6, 4 * len(metrics)), sharex=True
    )
    fig.suptitle("Gaussian Noise: Metrics vs SD for Each Model", fontsize=16)

    if len(metrics) == 1:
        axes = [axes]

    colors = {
        "Lecture": "tab:blue",
        "KLDivergence": "tab:orange",
        "L21": "tab:green",
    }
    import numpy as np

    for i, metric in enumerate(metrics):
        plot_df = df_g[
            ["NMF Method", "sd", f"perc_diff_to_no_noise_{metric}", f"{metric}_sd"]
        ]
        sns.lineplot(
            data=plot_df,
            x="sd",
            y=f"perc_diff_to_no_noise_{metric}",
            hue="NMF Method",
            marker="o",
            palette=colors,
            ax=axes[i],
        )
        axes[i].set_title(f"Metric: {metric}")
        # axes[i].set_ylim(
        #     min(plot_df[f"{metric}_mu"]) * 0.9, min(plot_df[f"{metric}_mu"]) * 1.1
        # )
        axes[i].set_ylabel("Rel. Perf. Change from Clean Data")
        axes[i].legend(title="Model", loc="best")

    axes[-1].set_xlabel("Noise SD")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved figure to {out_file}")


def plot_uniform(df, out_file="viz/uniform_noise.png"):
    df_u = df[
        (df["Noise Method"] == "uniform_random")
        & (df["Data Loader"] == "DataLoaderORL")
    ]
    if df_u.empty:
        print("No Uniform noise results found!")
        return

    models = df_u["NMF Method"].unique()
    fig, axes = plt.subplots(
        len(models), 1, figsize=(6, 4 * len(models)), sharex=True, sharey=True
    )
    fig.suptitle("ORL Dataset - Uniform Noise: Metrics vs Variance for Each Model", fontsize=16)

    if len(models) == 1:
        axes = [axes]

    colors = {"ACC_mu": "tab:blue", "RRE_mu": "tab:orange", "NMI_mu": "tab:green"}

    for ax, model in zip(axes, models):
        subset = df_u[df_u["NMF Method"] == model]
        plot_df = subset.melt(
            id_vars=["var"],
            value_vars=["ACC_mu", "RRE_mu", "NMI_mu"],
            var_name="Metric",
            value_name="Value",
        )
        sns.lineplot(
            data=plot_df,
            x="var",
            y="Value",
            hue="Metric",
            marker="o",
            palette=colors,
            ax=ax,
        )
        ax.set_title(f"Model: {model}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.legend(title="Metric", loc="upper right")

    axes[-1].set_xlabel("Noise Variance")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved figure to {out_file}")


if __name__ == "__main__":
    df = load_results("results")
    print(df)
    save_to_csv(df, "results_summary.csv")
    for dataset in ["DataLoaderORL", "DataLoaderYale"]:
        plot_salt_and_pepper_noise_2(df, dataset)
        plot_gaussian_noise_2(df, dataset)
        plot_uniform_noise_2(df, dataset)
