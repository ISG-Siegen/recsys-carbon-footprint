import time

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.ticker as ticker


# mpl.use('macosx')


def figure1(data, save_path):
    x_1 = 'total_energy_consumed (total energy consumed, in Wh)'
    y_1 = 'recommender'
    x_2 = 'total_energy_consumed (total energy consumed, in Wh)'
    y_2 = 'dataset'

    fontsize = 18

    data.loc[leviathan_data['dataset'] == "Yelp-2018", 'dataset'] = "Yelp-2018 \n 3.3M Interactions"
    data.loc[leviathan_data['dataset'] == "MovieLens-1M", 'dataset'] = "MovieLens-1M \n 1M Interactions"
    data.loc[leviathan_data['dataset'] == "MovieLens-100K", 'dataset'] = "MovieLens-100K \n 100K Interactions"
    data.loc[leviathan_data['dataset'] == "Retailrocket", 'dataset'] = "Retailrocket \n 241K Interactions"
    data.loc[leviathan_data['dataset'] == "Hetrec-LastFM", 'dataset'] = "Hetrec-LastFM \n 53K Interactions"
    data.loc[leviathan_data['dataset'] == "Gowalla", 'dataset'] = "Gowalla \n 2M Interactions"
    data.loc[leviathan_data['dataset'] == "Amazon2018-Books", 'dataset'] = "Amazon2018-Books \n 1.7M Interactions"
    data.loc[leviathan_data[
                 'dataset'] == "Amazon2018-CDs-And-Vinyl", 'dataset'] = "Amazon2018-CDs-And-Vinyl \n 1.4M Interactions"
    data.loc[leviathan_data[
                 'dataset'] == "Amazon2018-Sports-And-Outdoors", 'dataset'] = "Amazon2018-Sports-And-Outdoors \n 1.5M Interactions"
    data.loc[leviathan_data[
                 'dataset'] == "Amazon2018-Electronics", 'dataset'] = "Amazon2018-Electronics \n 1.5M Interactions"
    data.loc[leviathan_data[
                 'dataset'] == "Amazon2018-Toys-And-Games", 'dataset'] = "Amazon2018-Toys-And-Games \n 1.7M Interactions"
    data.loc[
        leviathan_data['dataset'] == "MovieLens-Latest-Small", 'dataset'] = "MovieLens-Latest-Small \n 90K Interactions"

    # Constants for conversion (World Average)
    wh_to_co2 = 438  # gCO2 per kWh

    # Aggregate fit, predict and evaluate stages
    data = data.groupby(['dataset', 'recommender']).agg({
        'total_energy_consumed (total energy consumed, in Wh)': sum,
        'year': 'first',
        'processor_type': 'first',
        'prediction_task': 'first'}).reset_index()

    # Drop duplicate Baselines
    data.drop(data[data['recommender'] == 'SVD'].index, inplace=True)
    data.drop(data[data['recommender'] == 'UserKNN$^{LK}$'].index, inplace=True)
    data.drop(data[data['recommender'] == 'ItemKNN$^{LK}$'].index, inplace=True)
    data.drop(data[data['recommender'] == 'ItemKNN$^{RP}$'].index, inplace=True)
    data.drop(data[data['recommender'] == 'Popularity$^{RP}$'].index, inplace=True)
    data.drop(data[data['recommender'] == "Popularity$^{EL}$"].index, inplace=True)
    data.drop(data[data['recommender'] == "MultiVAE$^{EL}$"].index, inplace=True)
    data.drop(data[data['recommender'] == "BPR$^{EL}$"].index, inplace=True)


    # Sort the data by total energy consumed
    data.sort_values(by='total_energy_consumed (total energy consumed, in Wh)', inplace=True, ascending=False)

    # # Create the figure and subplots with shared x-axis
    fig, ax = plt.subplots(figsize=(9, 10))
    sns.color_palette("tab10")

    # sns.set_theme(style="whitegrid")

    # Plotting the bar plots using seaborn
    plt.rcParams.update({'font.size': 18})
    sns.boxplot(data=data, x=x_1, y=y_1, ax=ax, saturation=1, width=0.8)
    # sns.set(font_scale=1.1)

    # axs[0].set_title('Average Algorithm Energy Consumption on 12 Datasets (in kWh)')
    ax.set_xlabel('[measured] Average Energy Consumption \n on 12 Datasets (in kWh)', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_ylabel('Recommender', fontsize=fontsize)

    # Add a second x-axis to show CO2 emissions
    x_1_2 = ax.twiny()
    # x_2_2 = axs[1].twiny()

    # Set the x-axis limits to match the original x-axis
    x_1_2_min, x_1_2_max = ax.get_xlim()
    # x_2_2_min, x_2_2_max = axs[1].get_xlim()

    # Convert the x-axis scale to CO2 emissions and set the label
    x_1_2.set_xlim(x_1_2_min * wh_to_co2, x_1_2_max * wh_to_co2)

    # Set the label for the second x-axis
    x_1_2.set_xlabel('[estimated] Grams of CO2 Equivalents \n Emitted (in gCO2e)', fontsize=fontsize)

    # x_2_2.set_xlabel('[estimated] Grams of CO2 Equivalents Emitted (in gCO2e)')

    algo_average = data.copy()
    algo_average = algo_average['total_energy_consumed (total energy consumed, in Wh)'].mean()

    color_palette = sns.color_palette('tab10')
    color = color_palette[0]

    ax.axvline(x=algo_average.mean(),  # Line on x = 2
               ymin=0,  # Bottom of the plot
               ymax=1,  # Top of the plot
               color='black',
               linewidth=1.1)

    ax.axvline(x=algo_average.mean(),  # Line on x = 2
               ymin=0,  # Bottom of the plot
               ymax=1,  # Top of the plot
               color=color,
               linewidth=1)

    # Adding text on the x-axis to indicate the position of the line
    ax.text(algo_average,
            ax.get_ylim()[0],
            'Average={}kWh'.format(round(algo_average, 2)),
            verticalalignment='bottom',
            horizontalalignment='left')

    ax.grid(True, which='both', axis='x')
    # axs[1].grid(True, which='both', axis='x')

    result = (
        data.groupby('recommender')['total_energy_consumed (total energy consumed, in Wh)']
        .agg(['mean', 'median', 'min', 'max'])
        .round(4)
        .rename(columns={
            'mean': 'Average (Wh)',
            'median': 'Median (Wh)',
            'min': 'Minimum (Wh)',
            'max': 'Maximum (Wh)'
        })
    )

    print(result.to_latex())

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()
    plt.close()


def figure1_2(data, save_path):
    plt.rcParams.update({'font.size': 18})
    x_1 = 'total_energy_consumed (total energy consumed, in Wh)'
    y_1 = 'recommender'
    x_2 = 'total_energy_consumed (total energy consumed, in Wh)'
    y_2 = 'dataset'

    data.loc[leviathan_data['dataset'] == "Yelp-2018", 'dataset'] = "Yelp-2018 \n 3.3M Interactions"
    data.loc[leviathan_data['dataset'] == "MovieLens-1M", 'dataset'] = "MovieLens-1M \n 1M Interactions"
    data.loc[leviathan_data['dataset'] == "MovieLens-100K", 'dataset'] = "MovieLens-100K \n 100K Interactions"
    data.loc[leviathan_data['dataset'] == "Retailrocket", 'dataset'] = "Retailrocket \n 241K Interactions"
    data.loc[leviathan_data['dataset'] == "Hetrec-LastFM", 'dataset'] = "Hetrec-LastFM \n 53K Interactions"
    data.loc[leviathan_data['dataset'] == "Gowalla", 'dataset'] = "Gowalla \n 2M Interactions"
    data.loc[leviathan_data['dataset'] == "Amazon2018-Books", 'dataset'] = "AMZ-Books \n 1.7M Interactions"
    data.loc[leviathan_data[
                 'dataset'] == "Amazon2018-CDs-And-Vinyl", 'dataset'] = "AMZ-CDs-And-Vinyl \n 1.4M Interactions"
    data.loc[leviathan_data[
                 'dataset'] == "Amazon2018-Sports-And-Outdoors", 'dataset'] = "AMZ-Sports-And-Outdoors \n 1.5M Interactions"
    data.loc[leviathan_data[
                 'dataset'] == "Amazon2018-Electronics", 'dataset'] = "AMZ-Electronics \n 1.5M Interactions"
    data.loc[leviathan_data[
                 'dataset'] == "Amazon2018-Toys-And-Games", 'dataset'] = "AMZ-Toys-And-Games \n 1.7M Interactions"
    data.loc[
        leviathan_data['dataset'] == "MovieLens-Latest-Small", 'dataset'] = "MovieLens-Latest-Small \n 90K Interactions"

    data.drop(data[data['dataset'] == 'Amazon2018-Clothing-Shoes-And-Jewelry'].index, inplace=True)

    # Constants for conversion (World Average)
    wh_to_co2 = 438  # gCO2 per kWh

    # Aggregate fit, predict and evaluate stages
    data = data.groupby(['dataset', 'recommender']).agg({
        'total_energy_consumed (total energy consumed, in Wh)': sum,
        'year': 'first',
        'processor_type': 'first',
        'prediction_task': 'first'}).reset_index()

    # Drop duplicate Baselines
    data.drop(data[data['recommender'] == 'SVD'].index, inplace=True)
    data.drop(data[data['recommender'] == 'UserKNN$^{LK}$'].index, inplace=True)
    data.drop(data[data['recommender'] == 'ItemKNN$^{LK}$'].index, inplace=True)
    data.drop(data[data['recommender'] == 'ItemKNN$^{RP}$'].index, inplace=True)
    data.drop(data[data['recommender'] == 'Popularity$^{RP}$'].index, inplace=True)
    data.drop(data[data['recommender'] == "Popularity$^{EL}$"].index, inplace=True)
    data.drop(data[data['recommender'] == "MultiVAE$^{EL}$"].index, inplace=True)
    data.drop(data[data['recommender'] == "BPR$^{EL}$"].index, inplace=True)

    # Sort the data by total energy consumed
    data.sort_values(by='total_energy_consumed (total energy consumed, in Wh)', inplace=True, ascending=False)

    # # Create the figure and subplots with shared x-axis
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.color_palette("tab10")

    # sns.set_theme(style="whitegrid")

    # Plotting the bar plots using seaborn
    # sns.boxplot(data=data, x=x_1, y=y_1, ax=axs[0], saturation=1)
    sns.boxplot(data=data, x=x_2, y=y_2, ax=ax, saturation=1)

    # fig.suptitle("Measured Averaged Energy Consumption and Estimated Emitted gCO2e for 16 Algorithms and 12 Datasets")

    # axs[0].set_title('Average Algorithm Energy Consumption on 12 Datasets (in kWh)')
    # axs[0].set_xlabel('[measured] Average Energy Consumption on 12 Datasets (in kWh)')
    # axs[0].set_ylabel('Recommender')

    # axs[1].set_title('Average Dataset Energy Consuption on 16 Algorithms (in kWh)')
    ax.set_xlabel('[measured] Average Energy Consumption \n on 19 Models (in kWh)')
    xlabel = plt.gca().get_xaxis().get_label()
    xlabel.set_position((0.55, 0))
    ax.set_ylabel('Dataset')

    # Add a second x-axis to show CO2 emissions
    # x_1_2 = axs[0].twiny()
    x_2_2 = ax.twiny()

    # Set the x-axis limits to match the original x-axis
    # x_1_2_min, x_1_2_max = axs[0].get_xlim()
    x_2_2_min, x_2_2_max = ax.get_xlim()

    # Convert the x-axis scale to CO2 emissions and set the label
    # x_1_2.set_xlim(x_1_2_min * wh_to_co2, x_1_2_max * wh_to_co2)
    x_2_2.set_xlim(x_2_2_min * wh_to_co2, x_2_2_max * wh_to_co2)

    # Set the label for the second x-axis
    # x_1_2.set_xlabel('[estimated] Grams of CO2 Equivalents Emitted (in gCO2e)')
    x_2_2.set_xlabel('[estimated] Grams of CO2 Equivalents \n Emitted (in gCO2e)')

    algo_average = data.copy()
    # algo_average.drop(data[data['recommender'] == 'ItemKNN$^{RB}$'].index, inplace=True)
    # algo_average.drop(data[data['recommender'] == 'Popularity$^{RB}$'].index, inplace=True)
    # algo_average.drop(data[data['recommender'] == 'ImplicitMF'].index, inplace=True)
    algo_average = algo_average['total_energy_consumed (total energy consumed, in Wh)'].mean()

    color_palette = sns.color_palette('tab10')
    color = color_palette[0]

    # axs[0].axvline(x=algo_average.mean(),  # Line on x = 2
    #                ymin=0,  # Bottom of the plot
    #                ymax=1,  # Top of the plot
    #                color='black',
    #                linewidth=1.1)
    #
    # axs[0].axvline(x=algo_average.mean(),  # Line on x = 2
    #                ymin=0,  # Bottom of the plot
    #                ymax=1,  # Top of the plot
    #                color=color,
    #                linewidth=1)

    ax.axvline(x=algo_average.mean(),  # Line on x = 2
               ymin=0,  # Bottom of the plot
               ymax=1,  # Top of the plot
               color='black',
               linewidth=1.1)

    ax.axvline(x=data['total_energy_consumed (total energy consumed, in Wh)'].mean(),  # Line on x = 2
               ymin=0,  # Bottom of the plot
               ymax=1,  # Top of the plot
               color=color,
               linewidth=1)

    # Adding text on the x-axis to indicate the position of the line
    # axs[0].text(algo_average,
    #             axs[0].get_ylim()[0],
    #             'Average={}kWh'.format(round(algo_average, 2)),
    #             verticalalignment='bottom',
    #             horizontalalignment='left')

    # Adding text on the x-axis to indicate the position of the line
    ax.text(data['total_energy_consumed (total energy consumed, in Wh)'].mean(),
            ax.get_ylim()[0],
            'Average={}kWh'.format(round(data['total_energy_consumed (total energy consumed, in Wh)'].mean(), 2)),
            verticalalignment='bottom',
            horizontalalignment='left')

    # axs[0].grid(True, which='both', axis='x')
    ax.grid(True, which='both', axis='x')
    ax.xaxis.set_major_locator(MultipleLocator(1))

    result = (
        data.groupby('dataset')['total_energy_consumed (total energy consumed, in Wh)']
        .agg(['mean', 'median', 'min', 'max'])
        .round(4)
        .rename(columns={
            'mean': 'Average (Wh)',
            'median': 'Median (Wh)',
            'min': 'Minimum (Wh)',
            'max': 'Maximum (Wh)'
        })
    )
    print(result.to_latex())

    #plt.subplots_adjust(left=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()
    plt.close()


def figure2(data, save_path):
    # sns.set_context("paper", font_scale=1.4)
    # sns.set(rc={"font.size": 1.4})
    plt.rcParams.update({'font.size': 13.15})

    data = data.groupby(['dataset', 'recommender', 'pc']).agg({
        'total_energy_consumed (total energy consumed, in Wh)': 'sum',
        'duration': 'sum',
        'year': 'first',
        'processor_type': 'first',
        'prediction_task': 'first'}).reset_index()

    data = data[data['dataset'].isin(mac_studio_datasets)]
    data = data[data['recommender'].isin(mac_studio_recommenders)]

    data.loc[data['pc'] == "old_pc", 'pc'] = "Legacy Workstation"
    data.loc[data['pc'] == "leviathan", 'pc'] = "Modern Workstation I"
    data.loc[data['pc'] == "macbook", 'pc'] = "MacBook Pro"
    data.loc[data['pc'] == "macStudio", 'pc'] = "Mac Studio"
    data.loc[data['pc'] == "workstation", 'pc'] = "Modern Workstation II"

    data.drop(data[data['pc'] == '2013 Workstation'].index, inplace=True)

    data.loc[data['processor_type'] == "CPU", 'processor_type'] = "Traditional"
    data.loc[data['processor_type'] == "GPU", 'processor_type'] = "Deep Learning"

    data['Executed on:'] = data['pc'] + ' / ' + data['processor_type'].astype(str)

    data['duration'] = data['duration'] / 3600  # Convert to hours

    # Constants for conversion
    wh_to_co2 = 438  # gCO2 per Wh

    # Desired dimensions
    desired_width = 3  # inches
    desired_height = 5  # inches
    #
    # # Set aspect to the ratio of width to height
    aspect_ratio = desired_width / desired_height

    # Rearrange the order of the data to bring the "2013 Workstation" plot to the foreground
    data['Computer / Algorithm Type'] = pd.Categorical(data['Executed on:'], [
        "MacBook Pro / Traditional",
        "MacBook Pro / Deep Learning",
        "Mac Studio / Traditional",
        "Mac Studio / Deep Learning",
        "Modern Workstation I / Traditional",
        "Modern Workstation I / Deep Learning",
        "Modern Workstation II / Traditional",
        "Modern Workstation II / Deep Learning", ])
    # "2013 Workstation (Traditional Algorithm)", ])

    # Define markers
    markers = ["x", "o", "x", "o", "x", "o", "x", "o"]
    order = [
        # "2013 Workstation (Traditional Algorithm)",
        "MacBook Pro / Traditional",
        "MacBook Pro / Deep Learning",
        "Mac Studio / Traditional",
        "Mac Studio / Deep Learning",
        "Modern Workstation I / Traditional",
        "Modern Workstation I / Deep Learning",
        "Modern Workstation II / Traditional",
        "Modern Workstation II / Deep Learning", ]

    # Create the lmplot with markers
    # plt.subplots_adjust(top=3)
    plt.figure(constrained_layout=True)
    g = sns.lmplot(data=data, x='duration', y='total_energy_consumed (total energy consumed, in Wh)',
                   hue='Computer / Algorithm Type', height=8, aspect=0.75,
                   scatter_kws={"s": 50},
                   markers=markers, hue_order=order,
                   palette=[color_macbook_pro, color_macbook_pro, color_mac_studio, color_mac_studio,
                            color_leviathan, color_leviathan, color_workstation, color_workstation])

    g.fig.set_size_inches(7, 5)

    # Get the legend handles and labels
    handles, labels = g.ax.get_legend_handles_labels()

    # # Bring the desired plot to the foreground by adjusting its z-order
    # for lh in handles:
    #     if lh.get_label() == '2013 Workstation (Traditional Algorithm)':
    #         lh.set_zorder(10)  # Set higher z-order for the desired plot

    # g.fig.suptitle("Energy Consumption and Runtime on Different Hardware Types")
    g.set_xlabels('Training, Prediction and Evaluation Time (in hours)')
    g.set_ylabels('[measured] Energy Consumed (in kWh)')

    # Adjust the figure to add a second y-axis
    ax1 = g.ax  # Get the existing axis
    ax2 = ax1.twinx()  # Create a new axis that shares the same x-axis

    # # Access the axes from the FacetGrid
    # ax = g.axes[0][0]  # Access the first (and perhaps only) axes
    # # Define new ticks
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))

    for ax in g.axes.flatten():
        ax.grid(True)

    # Convert the y-axis scale to CO2 emissions and set the label
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min * wh_to_co2, y1_max * wh_to_co2)
    ax2.set_ylabel('[estimated] Grams of CO2 Equivalents \n Emitted (in gCO2e)')

    sns.move_legend(g, "upper center", bbox_to_anchor=(0.53, 0.93), title="Computer / Algorithm Type",
                    labelspacing=0.25)

    leg = g._legend
    leg.set_frame_on(True)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.2)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(1)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


def figure3(data: pd.DataFrame, traditional_models, gpu_models, save_path):
    plt.rcParams.update({'font.size': 25})
    fontsize = 12
    # Normalize the names in lists
    traditional_models = [name.strip() for name in traditional_models]
    gpu_models = [name.strip() for name in gpu_models]

    duration_df = data.groupby(['dataset', 'recommender']).sum()[
        'total_energy_consumed (total energy consumed, in Wh)'].reset_index()
    performance_df = data.groupby(['dataset', 'recommender'])['NDCG@10'].mean().reset_index()
    scatterplot_df = pd.merge(duration_df, performance_df, on=['dataset', 'recommender'])

    mean_ndcg = scatterplot_df.groupby('dataset')['NDCG@10'].mean()
    scatterplot_df['normalized_NDCG'] = scatterplot_df.apply(lambda row: row['NDCG@10'] / mean_ndcg[row['dataset']],
                                                             axis=1)

    scatterplot_df = scatterplot_df.groupby('recommender')[
        ['normalized_NDCG', 'total_energy_consumed (total energy consumed, in Wh)']].mean().reset_index()
    scatterplot_df['recommender'] = scatterplot_df['recommender'].apply(lambda x: x.strip())
    recommender_to_number = {name: i + 1 for i, name in enumerate(scatterplot_df['recommender'].unique())}
    scatterplot_df['recommender_number'] = scatterplot_df['recommender'].map(recommender_to_number)

    # Assign colors and markers
    scatterplot_df['color'] = scatterplot_df['recommender'].apply(
        lambda x: color_cpu if x in gpu_models else color_gpu)

    plt.figure(figsize=(6, 6))

    sns.set_theme(style="whitegrid")
    ax1 = plt.gca()  # Get the current axis

    for idx, row in scatterplot_df.iterrows():
        marker = 'X' if row["recommender"] in traditional_models else 'o'
        plt.scatter(x=[row["total_energy_consumed (total energy consumed, in Wh)"]],
                    y=[row["normalized_NDCG"]],
                    color=row['color'],
                    marker=marker,
                    label=f'{row["recommender"]} ({row["recommender_number"]})', s=100)

    for i in range(scatterplot_df.shape[0]):
        x_offset = -0.02  # Adjust as needed
        y_offset = 0.00  # Adjust as needed
        label = str(scatterplot_df.iloc[i]['recommender_number'])

        # Custom adjustments for specific labels (check your plot for overlaps)
        if label == "12":  # Example for label "2"
            y_offset += -0.07
        elif label == "22":
            y_offset += -0.03 # Example for label "2"
        elif label == "10":  # Example for label "2"
            x_offset += 0.05
            y_offset += 0.03


        plt.text(scatterplot_df.iloc[i]['total_energy_consumed (total energy consumed, in Wh)'] + x_offset,
                 scatterplot_df.iloc[i]['normalized_NDCG'] + y_offset,
                 str(scatterplot_df.iloc[i]['recommender_number']),
                 horizontalalignment='right', size='medium', color='black', weight='semibold')

    plt.xlabel('[measured] Energy Consumed (in kWh)', fontsize=fontsize)
    plt.ylabel('Averaged And Normalized NDCG@10 Performance', fontsize=fontsize)
    # plt.title("Relation Between Energy Consumption and Performance of Recommenders")

    # Handling the first legend (color)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    first_legend = plt.legend(by_label.values(), by_label.keys(), title="Recommender", loc="lower center",
                              ncol=3, bbox_to_anchor=(0.5, -0.8), labelspacing=0.2)
    plt.gca().add_artist(first_legend)

    # Create second legend for shapes
    circle = mlines.Line2D([], [], color=color_gpu, marker='o', linestyle='None',
                           markersize=10, label='Deep Learning (Executed on GPU)')
    cross = mlines.Line2D([], [], color=color_cpu, marker='X', linestyle='None',
                          markersize=10, label='Traditional (Executed on CPU)')
    cross2 = mlines.Line2D([], [], color=color_gpu, marker='X', linestyle='None',
                           markersize=10, label='Traditional (Executed on GPU)')
    second_legend = plt.legend(handles=[circle, cross, cross2], loc='lower center', bbox_to_anchor=(0.66, 0.55),
                               title="Algorithm Type", labelspacing=0.5)
    second_legend.get_frame().set_linewidth(1.5)
    plt.gca().add_artist(second_legend)
    plt.subplots_adjust(bottom=0.29)

    # Create the second x-axis
    ax2 = ax1.twiny()
    ax1_xmin, ax1_xmax = ax1.get_xlim()
    ax2.set_xlim(ax1_xmin * 438, ax1_xmax * 438)
    ax2.set_xlabel('[estimated] Grams of CO2 Equivalents Emitted (in gCO2e)')
    ax2.grid(False)

    #plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def figure4(data, save_path):
    plt.rcParams.update({'font.size': 18})
    fontsize = 20

    x = 'total_energy_consumed (total energy consumed, in Wh)'
    y = 'recommender'
    # title = 'Average Algorithm Energy Consumption on 7 Datasets (in kWh)'

    data = data.groupby(['dataset', 'recommender']).agg({
        'total_energy_consumed (total energy consumed, in Wh)': sum,
        'year': 'first',
        'processor_type': 'first',
        'prediction_task': 'first'}).reset_index()

    data.rename(columns={'prediction_task': 'Prediction Task'}, inplace=True)

    data.sort_values(by='total_energy_consumed (total energy consumed, in Wh)', inplace=True, ascending=False)

    # Create the figure and subplots with shared x-axis
    fig, axs = plt.subplots(figsize=(9, 8))

    # Plotting the bar plots using seaborn
    sns.boxplot(data=data, x=x, y=y, ax=axs, hue='Prediction Task', whis=(0, 100),
                palette=[color_old_pc, color_rating_prediction], saturation=1)

    plt.legend(loc='center right')

    rating_prediciont_average = data[data['Prediction Task'] == 'Rating Prediction'][
        'total_energy_consumed (total energy consumed, in Wh)'].mean()
    ranking_prediciont_average = data[data['Prediction Task'] == 'Ranking Prediction'][
        'total_energy_consumed (total energy consumed, in Wh)'].mean()

    axs.axvline(x=rating_prediciont_average,  # Line on x = 2
                ymin=0,  # Bottom of the plot
                ymax=1,  # Top of the plot
                color=color_rating_prediction)

    axs.axvline(x=ranking_prediciont_average,  # Line on x = 2
                ymin=0,  # Bottom of the plot
                ymax=1,  # Top of the plot
                color=color_old_pc)

    axs.text(rating_prediciont_average,
             axs.get_ylim()[0] - 1,
             'Average={}kWh'.format(round(rating_prediciont_average, 2)),
             verticalalignment='top',
             horizontalalignment='left',
             color=color_rating_prediction)

    # Adding text on the x-axis to indicate the position of the line
    axs.text(ranking_prediciont_average,
             axs.get_ylim()[0],
             'Average={}kWh'.format(round(ranking_prediciont_average, 2)),
             verticalalignment='bottom',
             horizontalalignment='left',
             color=color_old_pc)

    ax2 = axs.twiny()
    ax1_xmin, ax1_xmax = axs.get_xlim()
    ax2.set_xlim(ax1_xmin * 438, ax1_xmax * 438)
    ax2.set_xlabel('[estimated] Grams of CO2 Equivalents \n Emitted (in gCO2e)')

    axs.set_xlabel('[measured] Energy Consumed (in kWh)')
    axs.set_ylabel('Recommender')

    axs.grid(True, which='both', axis='x')

    # plt.title(title)
    plt.tight_layout()
    # Saving the plot to the specified path
    plt.savefig(save_path, dpi=600)
    # Displaying the plot
    plt.show()
    # Closing the plot to free up memory
    plt.close()


def figure5(data, save_path):
    # Ensure a copy is used to avoid SettingWithCopyWarning
    data = data.copy()
    plt.rcParams.update({'font.size': 18})

    # Find indices where processor_type is 'CPU' and year is 2023
    indices_to_drop = data[
        (data['processor_type'] == 'CPU') & (data['year'] == '2023')].index

    # Drop these rows using the indices
    data = data.drop(indices_to_drop)

    data = data.groupby(['dataset', 'recommender']).agg({
        'total_energy_consumed (total energy consumed, in Wh)': sum,
        'year': 'first',
        'processor_type': 'first'}).reset_index()

    # Constants for CO2 emissions per Wh for different years and regions
    co2_emissions = {
        '2013': {'World \n Average': 486, 'North \n America': 427, 'Europe': 348, 'Asia': 604, 'Sweden': 5},
        '2023': {'World \n Average': 438, 'North \n America': 337, 'Europe': 297, 'Asia': 535, 'Sweden': 45}
    }

    # Adding energy consumption directly to avoid SettingWithCopyWarning
    data['Energy Consumed'] = data['total_energy_consumed (total energy consumed, in Wh)']

    # Calculate CO2 emissions for each region and year
    for year, regions in co2_emissions.items():
        for region, co2_value in regions.items():
            data.loc[data['year'] == year, region] = data['Energy Consumed'] * co2_value
            data.loc[data[
                         'year'] == year, 'Year_Processor'] = f"{year} {'Traditional (CPU)' if year == '2013' else 'Deep Learning (GPU)'} Algorithm \n Executed on {year} Workstation"

    # Melt the data for plotting
    melted_data = data.melt(id_vars=['Year_Processor', 'recommender'], value_vars=list(co2_emissions['2013'].keys()),
                            var_name='Region', value_name='CO2 Emissions (in g)')

    # # Sort the data by total energy consumed
    # melted_data.sort_values(by='Year_Processor', inplace=True, ascending=True)

    # Define hue order and palette explicitly
    hue_order = ['2023 Deep Learning (GPU) Algorithm \n Executed on 2023 Workstation',
                 '2013 Traditional (CPU) Algorithm \n Executed on 2013 Workstation']

    # Create the plot
    fig, axs = plt.subplots(figsize=(10, 8))

    # Draw the bar plot
    barplot = sns.barplot(data=melted_data, x='CO2 Emissions (in g)', y='Region', hue='Year_Processor', errorbar=None,
                          hue_order=hue_order,
                          palette=[color_leviathan, color_old_pc])

    # Ensure the grid is behind all other plot elements
    axs.set_axisbelow(True)  # This forces the grid to be drawn below plot elements

    # Set the zorder of the grid lines to -1 to ensure they are behind the bars
    axs.grid(True, which='both', axis='x', zorder=-1)

    # Now, manually set the zorder of the bars higher than the grid
    for bar in barplot.patches:
        bar.set_zorder(1)

    # plt.title('CO2 Emissions by Region and Algorithm Type')
    plt.xlabel('[estimated] Grams of CO2 Equivalents Emitted (in gCO2e)')
    plt.ylabel('Region')
    plt.legend(title='Algorithm Type/Year')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()


def figure6(data, save_path):
    data = data[data['stage'] != 'evaluate']
    data = data[data['stage'] != 'predict']
    data.sort_values(by='total_energy_consumed (total energy consumed, in Wh)', inplace=True, ascending=False)
    fig, ax = plt.subplots()
    ax.axvline(x=data[data['stage'] == 'fit']['total_energy_consumed (total energy consumed, in Wh)'].mean(),
               # Line on x = 2
               ymin=0,  # Bottom of the plot
               ymax=1,  # Top of the plot
               color='black',
               linewidth=1.1)
    # Boxplot for the energy consumption of the different recommenders and fit, predict and evaluate stages
    sns.boxplot(data=data, x='total_energy_consumed (total energy consumed, in Wh)', y='recommender', hue='stage')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def figure7(data, save_path):
    data = data[data['stage'] != 'evaluate']
    data = data[data['stage'] != 'fit']
    data.sort_values(by='total_energy_consumed (total energy consumed, in Wh)', inplace=True, ascending=False)
    fig, ax = plt.subplots()
    ax.axvline(x=data[data['stage'] == 'predict']['total_energy_consumed (total energy consumed, in Wh)'].mean(),
               # Line on x = 2
               ymin=0,  # Bottom of the plot
               ymax=1,  # Top of the plot
               color='black',
               linewidth=1.1)
    # Boxplot for the energy consumption of the different recommenders and fit, predict and evaluate stages
    sns.boxplot(data=data, x='total_energy_consumed (total energy consumed, in Wh)', y='recommender', hue='stage')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def figure8(data, save_path):
    wh_to_co2 = 438
    data = data[data['stage'] != 'evaluate']

    # Drop duplicate Baselines
    data.drop(data[data['recommender'] == 'SVD'].index, inplace=True)
    data.drop(data[data['recommender'] == 'UserKNN$^{LK}$'].index, inplace=True)
    data.drop(data[data['recommender'] == 'ItemKNN$^{LK}$'].index, inplace=True)
    data.drop(data[data['recommender'] == 'ItemKNN$^{RP}$'].index, inplace=True)
    data.drop(data[data['recommender'] == 'Popularity$^{RP}$'].index, inplace=True)
    data.drop(data[data['recommender'] == "Popularity$^{EL}$"].index, inplace=True)
    data.drop(data[data['recommender'] == "MultiVAE$^{EL}$"].index, inplace=True)
    data.drop(data[data['recommender'] == "BPR$^{EL}$"].index, inplace=True)

    # Calculate the total energy consumed for each recommender (sum of "fit" and "predict")
    total_energy = data.groupby('recommender')['total_energy_consumed (total energy consumed, in Wh)'].sum().reset_index()

    # Sort the recommenders by total energy consumed (ascending or descending order)
    total_energy_sorted = total_energy.sort_values('total_energy_consumed (total energy consumed, in Wh)', ascending=False)

    # Get the order of recommenders based on the sorted total energy consumption
    ordered_recommenders = total_energy_sorted['recommender'].tolist()

    # Calculate the average total energy consumed for each stage (fit and predict)
    average_fit = data[data['stage'] == 'fit']['total_energy_consumed (total energy consumed, in Wh)'].mean()
    average_predict = data[data['stage'] == 'predict']['total_energy_consumed (total energy consumed, in Wh)'].mean()
    data.loc[data['stage'] == 'fit', 'stage'] = 'train'


    fig, axs = plt.subplots(figsize=(6, 6))

    # Add vertical lines for the average energy consumed during "fit" and "predict"
    plt.axvline(x=average_fit, color=sns.color_palette('tab10')[0], linestyle='-')
    plt.axvline(x=average_predict, color=sns.color_palette('tab10')[1], linestyle='-')

    data = data.rename(columns={"recommender": "Recommender", "stage": "Phase"})

    # Create the boxplot, with 'recommender' on the y-axis and 'total_energy_consumed' on the x-axis
    sns.boxplot(data=data, y='Recommender', x='total_energy_consumed (total energy consumed, in Wh)', hue='Phase',
                order=ordered_recommenders, palette=[color_fit, color_predict])


    ax = plt.gca()  # Get current axis

    # if ax.legend_:
    #     ax.legend_.remove()

    ax.set_xlabel('[measured] Average Energy Consumption \n across 12 Datasets (in kWh)')

    ax.text(average_fit+0.85,
            ax.get_ylim()[0]-1.5,  # Position just below the plot
            f'Avg={round(average_fit, 2)} kWh',  # Convert to kWh for label
            verticalalignment='top',
            horizontalalignment='center',
            fontsize=10, color=sns.color_palette('tab10')[0])

    ax.text(average_predict+0.85,
            ax.get_ylim()[0]-0.8,  # Position just below the plot
            f'Avg={round(average_predict, 2)} kWh',  # Convert to kWh for label
            verticalalignment='top',
            horizontalalignment='center',
            fontsize=10, color=sns.color_palette('tab10')[1])

    # Create a second X-axis at the top to show CO2 emissions in grams (gCO2e)
    x_1_2 = ax.twiny()

    # Get the current limits of the primary X-axis (total energy in Wh)
    x_1_2_min, x_1_2_max = ax.get_xlim()

    # Set the limits of the second X-axis by converting Wh to gCO2e
    x_1_2.set_xlim(x_1_2_min * wh_to_co2, x_1_2_max * wh_to_co2)

    # Set the label for the second X-axis (gCO2e)
    x_1_2.set_xlabel('[estimated] Grams of CO2 Equivalents \n Emitted (in gCO2e)')

    ax.grid(True, which='both', axis='x')

    # Adding text on the x-axis to indicate the position of the line
    # Customize plot
    #plt.title('Energy Consumption by Recommender (Fit vs Predict)')
    #plt.xlabel('[measured] Average Energy Consumption on 12 Datasets (in kWh)')
    plt.ylabel('Recommender')
    #plt.legend(title='Stage', loc='upper left')

    #plt.legend(title='Stage', loc='center left', bbox_to_anchor=(0.7, 0.16), fontsize=12)

    # Display the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()


cpu_recommenders = ["ItemKNN$^{LK}$", "UserKNN$^{LK}$", "ImplicitMF$^{LK}$", "Popularity$^{RP}$",
                    "ItemKNN$^{RP}$", "NMF$^{RP}$", "SVD$^{RP}$", "AMF$^{EL}$", "Popularity$^{EL}$", "BPR$^{EL}$"]


traditional_recommenders = ["ItemKNN$^{LK}$", "UserKNN$^{LK}$", "ImplicitMF$^{LK}$", "Popularity$^{RP}$",
                            "ItemKNN$^{RP}$", "BPR$^{RB}$", "NMF$^{RP}$", "SVD$^{RP}$", "ItemKNN$^{RB}$",
                            "Popularity$^{RB}$", "AMF$^{EL}$", "Popularity$^{EL}$", "BPR$^{EL}$"]

mac_studio_datasets = ['Hetrec-LastFM', 'MovieLens-100K', 'MovieLens-1M', 'MovieLens-Latest-Small',
                       "Amazon2018-Electronics", "Amazon2018-Toys-And-Games",
                       "Amazon2018-Sports-And-Outdoors"]

mac_studio_recommenders = ["Popularity$^{RP}$", "ItemItem$^{LK}$", "UserUser$^{LK}$", "ImplicitMF$^{LK}$",
                           "SVD$^{RP}$", "NMF$^{RP}$", "ItemKNN$^{RP}$",
                           "ItemKNN$^{RB}$", "BPR$^{RB}$", "NeuMF$^{RB}$", "MultiVAE$^{RB}$", "RecVAE$^{RB}$"]

# Read the data
old_pc_rating_prediction_data = pd.read_csv('experiment_logs_rating_prediction_old_pc/mapped_logs_overall_old_pc.csv')
old_pc_ranking_prediction_data = pd.read_csv('experiment_logs_ranking_prediction_old_pc/mapped_logs_overall_old_pc.csv')
leviathan_data = pd.read_csv('experiment_logs_leviathan/mapped_logs_overall_leviathan.csv')
macStudio_data = pd.read_csv('experiment_logs_macStudio/mapped_logs_overall_macStudio.csv')
macStudio_data_journal = pd.read_csv('experiment_logs_macStudio_journal/mapped_logs_overall_macStudio.csv')
macbook_data = pd.read_csv('experiment_logs_macbook/macbook.csv', sep=';')
macbook_data_journal = pd.read_csv('experiment_logs_macbook_journal/mapped_logs_overall_macbook.csv')
workstation_data = pd.read_csv('experiment_logs_workstation/mapped_logs_overall_workstation.csv')
workstation_data_journal = pd.read_csv('experiment_logs_workstation_journal/mapped_logs_overall_workstation_journal.csv')
leviathan_data_journal = pd.read_csv('experiment_logs_leviathan_journal/mapped_logs_overall_leviathan_journal.csv')

leviathan_data = pd.concat([leviathan_data, leviathan_data_journal], ignore_index=True)
workstation_data = pd.concat([workstation_data, workstation_data_journal], ignore_index=True)
macStudio_data = pd.concat([macStudio_data, macStudio_data_journal], ignore_index=True)
macbook_data = pd.concat([macbook_data, macbook_data_journal], ignore_index=True)


# Rename to avoid duplicate labels
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['dataset'] == "MovieLens-1M", 'dataset'] = "MovieLens-1M w/ Ratings"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['dataset'] == "MovieLens-100K", 'dataset'] = "MovieLens-10p0K w/ Ratings"
leviathan_data.loc[leviathan_data['recommender'] == "ItemItem", 'recommender'] = "ItemKNN$^{LK}$"
leviathan_data.loc[leviathan_data['recommender'] == "UserUser", 'recommender'] = "UserKNN$^{LK}$"
leviathan_data.loc[leviathan_data['recommender'] == "ItemKNNRP", 'recommender'] = "ItemKNN$^{RP}$"
leviathan_data.loc[leviathan_data['recommender'] == "PopScore", 'recommender'] = "Popularity$^{RP}$"
leviathan_data.loc[leviathan_data['recommender'] == "Pop", 'recommender'] = "Popularity$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "ItemKNN", 'recommender'] = "ItemKNN$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "SVD", 'recommender'] = "SVD$^{RP}$"
leviathan_data.loc[leviathan_data['recommender'] == "NMF", 'recommender'] = "NMF$^{RP}$"
leviathan_data.loc[leviathan_data['recommender'] == "BPR", 'recommender'] = "BPR$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "DGCF", 'recommender'] = "DGCF$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "DMF", 'recommender'] = "DMF$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "LightGCN", 'recommender'] = "LightGCN$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "MacridVAE", 'recommender'] = "MacridVAE$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "MultiVAE", 'recommender'] = "MultiVAE$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "NAIS", 'recommender'] = "NAIS$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "NCL", 'recommender'] = "NCL$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "NeuMF", 'recommender'] = "NeuMF$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "NGCF", 'recommender'] = "NGCF$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "RecVAE", 'recommender'] = "RecVAE$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "SGL", 'recommender'] = "SGL$^{RB}$"
leviathan_data.loc[leviathan_data['recommender'] == "ImplicitMF", 'recommender'] = "ImplicitMF$^{LK}$"
leviathan_data.loc[leviathan_data['recommender'] == "AMF", 'recommender'] = "AMF$^{EL}$"
leviathan_data.loc[leviathan_data['recommender'] == "MultiDAEEL", 'recommender'] = "MultiDAE$^{EL}$"
leviathan_data.loc[leviathan_data['recommender'] == "MultiVAEEL", 'recommender'] = "MultiVAE$^{EL}$"
leviathan_data.loc[leviathan_data['recommender'] == "BPRMF_batch", 'recommender'] = "BPR$^{EL}$"
leviathan_data.loc[leviathan_data['recommender'] == "MostPop", 'recommender'] = "Popularity$^{EL}$"

old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "ItemItem", 'recommender'] = "ItemKNN$^{LK}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "UserUser", 'recommender'] = "UserKNN$^{LK}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "ItemKNNRP", 'recommender'] = "ItemKNN$^{RP}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "PopScore", 'recommender'] = "Popularity$^{RP}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "Pop", 'recommender'] = "Popularity$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "ItemKNN", 'recommender'] = "ItemKNN$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "SVD", 'recommender'] = "SVD$^{RP}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "NMF", 'recommender'] = "NMF$^{RP}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "BPR", 'recommender'] = "BPR$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "DGCF", 'recommender'] = "DGCF$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "DMF", 'recommender'] = "DMF$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "LightGCN", 'recommender'] = "LightGCN$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "MacridVAE", 'recommender'] = "MacridVAE$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "MultiVAE", 'recommender'] = "MultiVAE$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "NAIS", 'recommender'] = "NAIS$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "NCL", 'recommender'] = "NCL$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "NeuMF", 'recommender'] = "NeuMF$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "NGCF", 'recommender'] = "NGCF$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "RecVAE", 'recommender'] = "RecVAE$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "SGL", 'recommender'] = "SGL$^{RB}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "ImplicitMF", 'recommender'] = "ImplicitMF$^{LK}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "FunkSVD", 'recommender'] = "FunkSVD$^{LK}$"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "BiasedMF", 'recommender'] = "BiasedMF$^{LK}$"

old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "ItemItem", 'recommender'] = "ItemKNN$^{LK}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "UserUser", 'recommender'] = "UserKNN$^{LK}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "ItemKNNRP", 'recommender'] = "ItemKNN$^{RP}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "PopScore", 'recommender'] = "Popularity$^{RP}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "Pop", 'recommender'] = "Popularity$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "ItemKNN", 'recommender'] = "ItemKNN$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "SVD", 'recommender'] = "SVD$^{RP}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "NMF", 'recommender'] = "NMF$^{RP}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "BPR", 'recommender'] = "BPR$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "DGCF", 'recommender'] = "DGCF$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "DMF", 'recommender'] = "DMF$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "LightGCN", 'recommender'] = "LightGCN$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "MacridVAE", 'recommender'] = "MacridVAE$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "MultiVAE", 'recommender'] = "MultiVAE$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "NAIS", 'recommender'] = "NAIS$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "NCL", 'recommender'] = "NCL$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "NeuMF", 'recommender'] = "NeuMF$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "NGCF", 'recommender'] = "NGCF$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "RecVAE", 'recommender'] = "RecVAE$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "SGL", 'recommender'] = "SGL$^{RB}$"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "ImplicitMF", 'recommender'] = "ImplicitMF$^{LK}$"

macStudio_data.loc[macStudio_data['recommender'] == "ItemItem", 'recommender'] = "ItemKNN$^{LK}$"
macStudio_data.loc[macStudio_data['recommender'] == "UserUser", 'recommender'] = "UserKNN$^{LK}$"
macStudio_data.loc[macStudio_data['recommender'] == "ItemKNNRP", 'recommender'] = "ItemKNN$^{RP}$"
macStudio_data.loc[macStudio_data['recommender'] == "PopScore", 'recommender'] = "Popularity$^{RP}$"
macStudio_data.loc[macStudio_data['recommender'] == "Pop", 'recommender'] = "Popularity$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "ItemKNN", 'recommender'] = "ItemKNN$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "SVD", 'recommender'] = "SVD$^{RP}$"
macStudio_data.loc[macStudio_data['recommender'] == "NMF", 'recommender'] = "NMF$^{RP}$"
macStudio_data.loc[macStudio_data['recommender'] == "BPR", 'recommender'] = "BPR$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "DGCF", 'recommender'] = "DGCF$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "DMF", 'recommender'] = "DMF$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "LightGCN", 'recommender'] = "LightGCN$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "MacridVAE", 'recommender'] = "MacridVAE$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "MultiVAE", 'recommender'] = "MultiVAE$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "NAIS", 'recommender'] = "NAIS$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "NCL", 'recommender'] = "NCL$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "NeuMF", 'recommender'] = "NeuMF$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "NGCF", 'recommender'] = "NGCF$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "RecVAE", 'recommender'] = "RecVAE$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "SGL", 'recommender'] = "SGL$^{RB}$"
macStudio_data.loc[macStudio_data['recommender'] == "ImplicitMF", 'recommender'] = "ImplicitMF$^{LK}$"
macStudio_data.loc[macStudio_data['recommender'] == "AMF", 'recommender'] = "AMF$^{EL}$"
macStudio_data.loc[macStudio_data['recommender'] == "MultiDAEEL", 'recommender'] = "MultiDAE$^{EL}$"
macStudio_data.loc[macStudio_data['recommender'] == "MultiVAEEL", 'recommender'] = "MultiVAE$^{EL}$"
macStudio_data.loc[macStudio_data['recommender'] == "BPRMF_batch", 'recommender'] = "BPR$^{EL}$"
macStudio_data.loc[macStudio_data['recommender'] == "MostPop", 'recommender'] = "Popularity$^{EL}$"

macbook_data.loc[macbook_data['recommender'] == "ItemItem", 'recommender'] = "ItemKNN$^{LK}$"
macbook_data.loc[macbook_data['recommender'] == "UserUser", 'recommender'] = "UserKNN$^{LK}$"
macbook_data.loc[macbook_data['recommender'] == "ItemKNNRP", 'recommender'] = "ItemKNN$^{RP}$"
macbook_data.loc[macbook_data['recommender'] == "PopScore", 'recommender'] = "Popularity$^{RP}$"
macbook_data.loc[macbook_data['recommender'] == "Pop", 'recommender'] = "Popularity$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "ItemKNN", 'recommender'] = "ItemKNN$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "SVD", 'recommender'] = "SVD$^{RP}$"
macbook_data.loc[macbook_data['recommender'] == "NMF", 'recommender'] = "NMF$^{RP}$"
macbook_data.loc[macbook_data['recommender'] == "BPR", 'recommender'] = "BPR$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "DGCF", 'recommender'] = "DGCF$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "DMF", 'recommender'] = "DMF$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "LightGCN", 'recommender'] = "LightGCN$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "MacridVAE", 'recommender'] = "MacridVAE$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "MultiVAE", 'recommender'] = "MultiVAE$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "NAIS", 'recommender'] = "NAIS$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "NCL", 'recommender'] = "NCL$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "NeuMF", 'recommender'] = "NeuMF$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "NGCF", 'recommender'] = "NGCF$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "RecVAE", 'recommender'] = "RecVAE$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "SGL", 'recommender'] = "SGL$^{RB}$"
macbook_data.loc[macbook_data['recommender'] == "ImplicitMF", 'recommender'] = "ImplicitMF$^{LK}$"
macbook_data.loc[macbook_data['recommender'] == "AMF", 'recommender'] = "AMF$^{EL}$"
macbook_data.loc[macbook_data['recommender'] == "MultiDAEEL", 'recommender'] = "MultiDAE$^{EL}$"
macbook_data.loc[macbook_data['recommender'] == "MultiVAEEL", 'recommender'] = "MultiVAE$^{EL}$"
macbook_data.loc[macbook_data['recommender'] == "BPRMF_batch", 'recommender'] = "BPR$^{EL}$"
macbook_data.loc[macbook_data['recommender'] == "MostPop", 'recommender'] = "Popularity$^{EL}$"

workstation_data.loc[workstation_data['recommender'] == "ItemItem", 'recommender'] = "ItemKNN$^{LK}$"
workstation_data.loc[workstation_data['recommender'] == "UserUser", 'recommender'] = "UserKNN$^{LK}$"
workstation_data.loc[workstation_data['recommender'] == "ItemKNNRP", 'recommender'] = "ItemKNN$^{RP}$"
workstation_data.loc[workstation_data['recommender'] == "PopScore", 'recommender'] = "Popularity$^{RP}$"
workstation_data.loc[workstation_data['recommender'] == "Pop", 'recommender'] = "Popularity$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "ItemKNN", 'recommender'] = "ItemKNN$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "SVD", 'recommender'] = "SVD$^{RP}$"
workstation_data.loc[workstation_data['recommender'] == "NMF", 'recommender'] = "NMF$^{RP}$"
workstation_data.loc[workstation_data['recommender'] == "BPR", 'recommender'] = "BPR$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "DGCF", 'recommender'] = "DGCF$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "DMF", 'recommender'] = "DMF$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "LightGCN", 'recommender'] = "LightGCN$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "MacridVAE", 'recommender'] = "MacridVAE$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "MultiVAE", 'recommender'] = "MultiVAE$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "NAIS", 'recommender'] = "NAIS$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "NCL", 'recommender'] = "NCL$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "NeuMF", 'recommender'] = "NeuMF$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "NGCF", 'recommender'] = "NGCF$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "RecVAE", 'recommender'] = "RecVAE$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "SGL", 'recommender'] = "SGL$^{RB}$"
workstation_data.loc[workstation_data['recommender'] == "ImplicitMF", 'recommender'] = "ImplicitMF$^{LK}$"
workstation_data.loc[workstation_data['recommender'] == "AMF", 'recommender'] = "AMF$^{EL}$"
workstation_data.loc[workstation_data['recommender'] == "MultiDAEEL", 'recommender'] = "MultiDAE$^{EL}$"
workstation_data.loc[workstation_data['recommender'] == "MultiVAEEL", 'recommender'] = "MultiVAE$^{EL}$"
workstation_data.loc[workstation_data['recommender'] == "BPRMF_batch", 'recommender'] = "BPR$^{EL}$"
workstation_data.loc[workstation_data['recommender'] == "MostPop", 'recommender'] = "Popularity$^{EL}$"


# Add columns to further classify the data
old_pc_rating_prediction_data['pc'] = 'old_pc'
old_pc_rating_prediction_data['prediction_task'] = 'Rating Prediction'
old_pc_rating_prediction_data['year'] = '2013'
old_pc_rating_prediction_data['processor_type'] = 'CPU'

old_pc_ranking_prediction_data['pc'] = 'old_pc'
old_pc_ranking_prediction_data['prediction_task'] = 'Ranking Prediction'
old_pc_ranking_prediction_data['year'] = '2013'
old_pc_ranking_prediction_data['processor_type'] = 'CPU'

macbook_data['pc'] = 'macbook'
macbook_data['prediction_task'] = 'Ranking Prediction'
macbook_data['year'] = '2019'
macbook_data['processor_type'] = 'GPU'
macbook_data.loc[macbook_data['recommender'].isin(cpu_recommenders), 'processor_type'] = 'CPU'

macStudio_data['pc'] = 'macStudio'
macStudio_data['prediction_task'] = 'Ranking Prediction'
macStudio_data['year'] = '2019'
macStudio_data['processor_type'] = 'GPU'
macStudio_data.loc[macStudio_data['recommender'].isin(cpu_recommenders), 'processor_type'] = 'CPU'

leviathan_data['pc'] = 'leviathan'
leviathan_data['prediction_task'] = 'Ranking Prediction'
leviathan_data['year'] = '2023'
leviathan_data['processor_type'] = 'GPU'
leviathan_data.loc[leviathan_data['recommender'].isin(cpu_recommenders), 'processor_type'] = 'CPU'

workstation_data['pc'] = 'workstation'
workstation_data['prediction_task'] = 'Ranking Prediction'
workstation_data['year'] = '2023'
workstation_data['processor_type'] = 'GPU'
workstation_data.loc[workstation_data['recommender'].isin(cpu_recommenders), 'processor_type'] = 'CPU'

old_pc_data = pd.concat([old_pc_rating_prediction_data, old_pc_ranking_prediction_data], ignore_index=True)
combined_data = pd.concat([old_pc_data, leviathan_data], ignore_index=True)
combined_data = pd.concat([combined_data, macbook_data], ignore_index=True)
combined_data = pd.concat([combined_data, macStudio_data], ignore_index=True)
combined_data = pd.concat([combined_data, workstation_data], ignore_index=True)

combined_data = combined_data[combined_data['recommender'] != 'ProximityBagging']
old_pc_data = old_pc_data[old_pc_data['recommender'] != 'ProximityBagging']
old_pc_rating_prediction_data = old_pc_rating_prediction_data[
    old_pc_rating_prediction_data['recommender'] != 'ProximityBagging']
old_pc_ranking_prediction_data = old_pc_ranking_prediction_data[
    old_pc_ranking_prediction_data['recommender'] != 'ProximityBagging']
leviathan_data = leviathan_data[leviathan_data['recommender'] != 'ProximityBagging']

combined_data = combined_data[combined_data['recommender'] != "BPR$^{RB}$"]
old_pc_data = old_pc_data[old_pc_data['recommender'] != "BPR$^{RB}$"]
old_pc_rating_prediction_data = old_pc_rating_prediction_data[
    old_pc_rating_prediction_data['recommender'] != "BPR$^{RB}$"]
old_pc_ranking_prediction_data = old_pc_ranking_prediction_data[
    old_pc_ranking_prediction_data['recommender'] != "BPR$^{RB}$"]
workstation_data = workstation_data[workstation_data['recommender'] != "BPR$^{RB}$"]

leviathan_data["Algorithm Type"] = "Deep Learning"
leviathan_data.loc[leviathan_data['recommender'].isin(traditional_recommenders), 'Algorithm Type'] = 'Traditional Model'

leviathan_data['total_energy_consumed (total energy consumed, in Wh)'] = leviathan_data[
                                                                             'total_energy_consumed (total energy consumed, in Wh)'] / 1000
workstation_data['total_energy_consumed (total energy consumed, in Wh)'] = workstation_data[
                                                                               'total_energy_consumed (total energy consumed, in Wh)'] / 1000
old_pc_data['total_energy_consumed (total energy consumed, in Wh)'] = old_pc_data[
                                                                          'total_energy_consumed (total energy consumed, in Wh)'] / 1000
macbook_data['total_energy_consumed (total energy consumed, in Wh)'] = macbook_data[
                                                                           'total_energy_consumed (total energy consumed, in Wh)'] / 1000
macStudio_data['total_energy_consumed (total energy consumed, in Wh)'] = macStudio_data[
                                                                             'total_energy_consumed (total energy consumed, in Wh)'] / 1000
combined_data['total_energy_consumed (total energy consumed, in Wh)'] = combined_data[
                                                                            'total_energy_consumed (total energy consumed, in Wh)'] / 1000

color_palette = sns.color_palette('tab10')
color_leviathan = color_palette[0]
color_old_pc = color_palette[1]
color_macbook_pro = color_palette[2]
color_mac_studio = color_palette[4]
color_workstation = color_palette[5]
color_fit = color_palette[0]
color_predict = color_palette[1]

color_gpu = color_palette[3]
color_cpu = 'black'
color_rating_prediction = color_palette[7]

# figure1(leviathan_data, 'figures/figure1.svg')
figure1_2(leviathan_data, 'figures/figure1_2.svg')
# figure2(combined_data, 'figures/figure2.svg')
# figure3(leviathan_data, traditional_models=traditional_recommenders, gpu_models=cpu_recommenders,
#        save_path='figures/figure3.svg')
# figure4(old_pc_data, 'figures/figure4.svg')
# figure5(combined_data, 'figures/figure5.svg')
# figure6 is experimental
# figure6(leviathan_data, 'figures/figure6.pdf')
# figure7 is experimental
# figure7(leviathan_data, 'figures/figure7.pdf')
# figure8(leviathan_data, 'figures/figure8.svg')
