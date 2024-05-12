
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


colors = sns.color_palette("husl", 5)

#paths = ["./exp/experiment_w_10_s_10_seed_7123", "./exp/experiment_w_10_s_10_seed_1287", 
#         "./exp/experiment_w_10_s_10_seed_6372", "./exp/experiment_w_10_s_10_seed_2651", 
#         "./exp/experiment_w_10_s_10_seed_199"]

def plot_experiment(paths, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for path in paths:
        df = pd.read_csv(path)
        sns.lineplot(df['Wolves'], ax=axes[0])
        sns.lineplot(df['Sheep'], ax=axes[1])
        sns.lineplot(df['Grass'], ax=axes[2])

    plt.savefig(save_path)

def plot_avg_std(paths, save_path):
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        dfs.append(df)

    concatenated_df = pd.concat(dfs)
    mean_values = concatenated_df.groupby(concatenated_df.index).mean()
    std_values = concatenated_df.groupby(concatenated_df.index).std()

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    sns.lineplot(x=mean_values.index, y=mean_values['Wolves'], ax=axes[0], color=colors[0], label='Mean')
    axes[0].fill_between(mean_values.index, mean_values['Wolves'] - std_values['Wolves'], mean_values['Wolves'] + std_values['Wolves'], alpha=0.3, color=colors[0])
    sns.lineplot(x=mean_values.index, y=mean_values['Sheep'], ax=axes[1], color=colors[1], label='Mean')
    axes[1].fill_between(mean_values.index, mean_values['Sheep'] - std_values['Sheep'], mean_values['Sheep'] + std_values['Sheep'], alpha=0.3, color=colors[1])
    sns.lineplot(x=mean_values.index, y=mean_values['Grass'], ax=axes[2], color=colors[2], label='Mean')
    axes[2].fill_between(mean_values.index, mean_values['Grass'] - std_values['Grass'], mean_values['Grass'] + std_values['Grass'], alpha=0.3, color=colors[2])

    axes[0].set_title('Wolves')
    axes[0].set_ylabel('Count')
    axes[1].set_title('Sheep')
    axes[1].set_ylabel('Count')
    axes[2].set_title('Grass')
    axes[2].set_ylabel('Count')
    for ax in axes:
        ax.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig(save_path)