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
import numpy as np
import matplotlib.ticker as ticker

# Colors
color_palette = sns.color_palette('tab10')
color_leviathan = color_palette[0]
color_old_pc = color_palette[1]
color_macbook_pro = color_palette[2]
color_mac_studio = color_palette[4]
color_gpu = color_palette[3]
color_cpu = 'black'
color_rating_prediction = color_palette[7]


def figure1(data, save_path):
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
    data.drop(data[data['recommender'] == 'UserKNN (LensKit)'].index, inplace=True)
    data.drop(data[data['recommender'] == 'ItemKNN (LensKit)'].index, inplace=True)
    data.drop(data[data['recommender'] == 'ItemKNN (RecPack)'].index, inplace=True)
    data.drop(data[data['recommender'] == 'Popularity (RecPack)'].index, inplace=True)

    # Sort the data by total energy consumed
    data.sort_values(by='total_energy_consumed (total energy consumed, in Wh)', inplace=True, ascending=False)

    # # Create the figure and subplots with shared x-axis
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    sns.color_palette("tab10")

    # sns.set_theme(style="whitegrid")

    # Plotting the bar plots using seaborn
    sns.boxplot(data=data, x=x_1, y=y_1, ax=axs[0], saturation=1)
    sns.boxplot(data=data, x=x_2, y=y_2, ax=axs[1], saturation=1)

    fig.suptitle("Measured Averaged Energy Consumption and Estimated Emitted gCO2e for 16 Algorithms and 12 Datasets")

    # axs[0].set_title('Average Algorithm Energy Consumption on 12 Datasets (in kWh)')
    axs[0].set_xlabel('[measured] Average Energy Consumption on 12 Datasets (in kWh)')
    axs[0].set_ylabel('Recommender')

    # axs[1].set_title('Average Dataset Energy Consuption on 16 Algorithms (in kWh)')
    axs[1].set_xlabel('[measured] Average Energy Consumption on 16 Algorithms (in kWh)')
    axs[1].set_ylabel('Dataset')

    # Add a second x-axis to show CO2 emissions
    x_1_2 = axs[0].twiny()
    x_2_2 = axs[1].twiny()

    # Set the x-axis limits to match the original x-axis
    x_1_2_min, x_1_2_max = axs[0].get_xlim()
    x_2_2_min, x_2_2_max = axs[1].get_xlim()

    # Convert the x-axis scale to CO2 emissions and set the label
    x_1_2.set_xlim(x_1_2_min * wh_to_co2, x_1_2_max * wh_to_co2)
    x_2_2.set_xlim(x_2_2_min * wh_to_co2, x_2_2_max * wh_to_co2)

    # Set the label for the second x-axis
    x_1_2.set_xlabel('[estimated] Grams of CO2 Equivalents Emitted (in gCO2e)')
    x_2_2.set_xlabel('[estimated] Grams of CO2 Equivalents Emitted (in gCO2e)')

    algo_average = data.copy()
    # algo_average.drop(data[data['recommender'] == 'ItemKNN (RecBole)'].index, inplace=True)
    # algo_average.drop(data[data['recommender'] == 'Popularity (RecBole)'].index, inplace=True)
    # algo_average.drop(data[data['recommender'] == 'ImplicitMF'].index, inplace=True)
    algo_average = algo_average['total_energy_consumed (total energy consumed, in Wh)'].mean()

    color_palette = sns.color_palette('tab10')
    color = color_palette[0]

    axs[0].axvline(x=algo_average.mean(),  # Line on x = 2
                   ymin=0,  # Bottom of the plot
                   ymax=1,  # Top of the plot
                   color='black',
                   linewidth=1.1)

    axs[0].axvline(x=algo_average.mean(),  # Line on x = 2
                   ymin=0,  # Bottom of the plot
                   ymax=1,  # Top of the plot
                   color=color,
                   linewidth=1)

    axs[1].axvline(x=algo_average.mean(),  # Line on x = 2
                   ymin=0,  # Bottom of the plot
                   ymax=1,  # Top of the plot
                   color='black',
                   linewidth=1.1)

    axs[1].axvline(x=data['total_energy_consumed (total energy consumed, in Wh)'].mean(),  # Line on x = 2
                   ymin=0,  # Bottom of the plot
                   ymax=1,  # Top of the plot
                   color=color,
                   linewidth=1)

    # Adding text on the x-axis to indicate the position of the line
    axs[0].text(algo_average,
                axs[0].get_ylim()[0],
                'Average={}kWh'.format(round(algo_average, 2)),
                verticalalignment='bottom',
                horizontalalignment='left')

    # Adding text on the x-axis to indicate the position of the line
    axs[1].text(data['total_energy_consumed (total energy consumed, in Wh)'].mean(),
                axs[1].get_ylim()[0],
                'Average={}kWh'.format(round(data['total_energy_consumed (total energy consumed, in Wh)'].mean(), 2)),
                verticalalignment='bottom',
                horizontalalignment='left')

    axs[0].grid(True, which='both', axis='x')
    axs[1].grid(True, which='both', axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()
    plt.close()


def figure2(data, save_path):
    data = data.groupby(['dataset', 'recommender', 'pc']).agg({
        'total_energy_consumed (total energy consumed, in Wh)': 'sum',
        'duration': 'sum',
        'year': 'first',
        'processor_type': 'first',
        'prediction_task': 'first'}).reset_index()

    data = data[data['dataset'].isin(mac_studio_datasets)]
    data = data[data['recommender'].isin(mac_studio_recommenders)]

    data.loc[data['pc'] == "old_pc", 'pc'] = "2013 Workstation"
    data.loc[data['pc'] == "leviathan", 'pc'] = "2023 Workstation"
    data.loc[data['pc'] == "macbook", 'pc'] = "2020 MacBook Pro"
    data.loc[data['pc'] == "macStudio", 'pc'] = "2022 Mac Studio"

    data.drop(data[data['pc'] == '2013 Workstation'].index, inplace=True)

    data.loc[data['processor_type'] == "CPU", 'processor_type'] = "Traditional Algorithm"
    data.loc[data['processor_type'] == "GPU", 'processor_type'] = "Deep Learning Algorithm"

    data['Executed on:'] = data['pc'] + ' (' + data['processor_type'].astype(str) + ')'

    data['duration'] = data['duration'] / 3600  # Convert to hours

    # Constants for conversion
    wh_to_co2 = 438  # gCO2 per Wh

    # Desired dimensions
    desired_width = 5  # inches
    desired_height = 5  # inches

    # Set aspect to the ratio of width to height
    aspect_ratio = desired_width / desired_height

    # Rearrange the order of the data to bring the "2013 Workstation" plot to the foreground
    data['Executed on:'] = pd.Categorical(data['Executed on:'], [
        "2020 MacBook Pro (Traditional Algorithm)",
        "2020 MacBook Pro (Deep Learning Algorithm)",
        "2022 Mac Studio (Traditional Algorithm)",
        "2022 Mac Studio (Deep Learning Algorithm)",
        "2023 Workstation (Traditional Algorithm)",
        "2023 Workstation (Deep Learning Algorithm)", ])
    # "2013 Workstation (Traditional Algorithm)", ])

    # Define markers
    markers = ["x", "o", "x", "o", "x", "o"]
    order = [
        # "2013 Workstation (Traditional Algorithm)",
        "2020 MacBook Pro (Traditional Algorithm)",
        "2020 MacBook Pro (Deep Learning Algorithm)",
        "2022 Mac Studio (Traditional Algorithm)",
        "2022 Mac Studio (Deep Learning Algorithm)",
        "2023 Workstation (Traditional Algorithm)",
        "2023 Workstation (Deep Learning Algorithm)", ]

    # Create the lmplot with markers

    g = sns.lmplot(data=data, x='duration', y='total_energy_consumed (total energy consumed, in Wh)',
                   hue='Executed on:', height=5, aspect=aspect_ratio, markers=markers, hue_order=order,
                   palette=[color_macbook_pro, color_macbook_pro, color_mac_studio, color_mac_studio,
                            color_leviathan, color_leviathan])

    # Get the legend handles and labels
    handles, labels = g.ax.get_legend_handles_labels()

    # Bring the desired plot to the foreground by adjusting its z-order
    for lh in handles:
        if lh.get_label() == '2013 Workstation (Traditional Algorithm)':
            lh.set_zorder(10)  # Set higher z-order for the desired plot

    g.fig.suptitle("Energy Consumption and Runtime on Different Hardware Types")
    g.set_xlabels('Training, Prediction and Evaluation Time (in hours)')
    g.set_ylabels('[measured] Energy Consumed (in kWh)')

    # Adjust the figure to add a second y-axis
    ax1 = g.ax  # Get the existing axis
    ax2 = ax1.twinx()  # Create a new axis that shares the same x-axis

    for ax in g.axes.flatten():
        ax.grid(True)

    leg = g._legend
    leg.set_frame_on(True)  # Ensure that there is a frame (box)
    leg.get_frame().set_edgecolor('black')  # Set the edge color of the frame
    leg.get_frame().set_linewidth(0.2)  # Set the thickness of the frame
    leg.get_frame().set_facecolor('white')  # Set the face color of the frame
    leg.get_frame().set_alpha(1)  # Set the transparency of the frame

    # Convert the y-axis scale to CO2 emissions and set the label
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min * wh_to_co2, y1_max * wh_to_co2)
    ax2.set_ylabel('[estimated] Grams of CO2 Equivalents Emitted (in gCO2e)')

    # Move the legend to the top right corner of the plot area
    g._legend.set_bbox_to_anchor((0.9, 0.7))

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()


def figure3(data: pd.DataFrame, traditional_models, gpu_models, save_path):
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

    plt.figure(figsize=(14, 10))
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
        x_offset = -0.01  # Adjust as needed
        y_offset = 0.00  # Adjust as needed
        plt.text(scatterplot_df.iloc[i]['total_energy_consumed (total energy consumed, in Wh)'] + x_offset,
                 scatterplot_df.iloc[i]['normalized_NDCG'] + y_offset,
                 str(scatterplot_df.iloc[i]['recommender_number']),
                 horizontalalignment='right', size='medium', color='black', weight='semibold')

    plt.xlabel('[measured] Energy Consumed (in kWh)')
    plt.ylabel('Averaged And Normalized NDCG@10 Performance')
    plt.title("Relation Between Energy Consumption and Performance of Recommenders")

    # Handling the first legend (color)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    first_legend = plt.legend(by_label.values(), by_label.keys(), title="Recommender", loc="upper center",
                              bbox_to_anchor=(0.63, -0.07), ncol=4)
    plt.gca().add_artist(first_legend)

    # Create second legend for shapes
    circle = mlines.Line2D([], [], color=color_gpu, marker='o', linestyle='None',
                           markersize=10, label='Deep Learning Algorithms (Executed on GPU)')
    cross = mlines.Line2D([], [], color=color_cpu, marker='X', linestyle='None',
                          markersize=10, label='Traditional Algorithms (Executed on CPU)')
    cross2 = mlines.Line2D([], [], color=color_gpu, marker='X', linestyle='None',
                           markersize=10, label='Traditional Algorithms (Executed on GPU)')
    plt.legend(handles=[circle, cross, cross2], loc='upper center', bbox_to_anchor=(0.12, -0.07),
               title="Algorithm Type")
    plt.subplots_adjust(bottom=-0.6)

    # Create the second x-axis
    ax2 = ax1.twiny()
    ax1_xmin, ax1_xmax = ax1.get_xlim()
    ax2.set_xlim(ax1_xmin * 438, ax1_xmax * 438)
    ax2.set_xlabel('[estimated] Grams of CO2 Equivalents Emitted (in gCO2e)')
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()
    plt.close()


def figure4(data, save_path):
    x = 'total_energy_consumed (total energy consumed, in Wh)'
    y = 'recommender'
    title = 'Average Algorithm Energy Consumption on 7 Datasets (in kWh)'

    data = data.groupby(['dataset', 'recommender']).agg({
        'total_energy_consumed (total energy consumed, in Wh)': sum,
        'year': 'first',
        'processor_type': 'first',
        'prediction_task': 'first'}).reset_index()

    data.rename(columns={'prediction_task': 'Prediction Task'}, inplace=True)

    data.sort_values(by='total_energy_consumed (total energy consumed, in Wh)', inplace=True, ascending=False)

    # Create the figure and subplots with shared x-axis
    fig, axs = plt.subplots(figsize=(7, 5))

    # Plotting the bar plots using seaborn
    sns.boxplot(data=data, x=x, y=y, ax=axs, hue='Prediction Task', whis=(0, 100),
                palette=[color_old_pc, color_rating_prediction], saturation=1)

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
    ax2.set_xlabel('[estimated] Grams of CO2 Equivalents Emitted (in gCO2e)')

    axs.set_xlabel('[measured] Energy Consumed (in kWh)')
    axs.set_ylabel('Recommender')

    axs.grid(True, which='both', axis='x')

    plt.title(title)
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
        '2013': {'World Average': 486, 'North America': 427, 'Europe': 348, 'Asia': 604, 'Sweden': 5},
        '2023': {'World Average': 438, 'North America': 337, 'Europe': 297, 'Asia': 535, 'Sweden': 45}
    }

    # Adding energy consumption directly to avoid SettingWithCopyWarning
    data['Energy Consumed'] = data['total_energy_consumed (total energy consumed, in Wh)']

    # Calculate CO2 emissions for each region and year
    for year, regions in co2_emissions.items():
        for region, co2_value in regions.items():
            data.loc[data['year'] == year, region] = data['Energy Consumed'] * co2_value
            data.loc[data[
                         'year'] == year, 'Year_Processor'] = f"{year} {'Traditional (CPU)' if year == '2013' else 'Deep Learning (GPU)'} \n Algorithm Executed \n on {year} Hardware"

    # Melt the data for plotting
    melted_data = data.melt(id_vars=['Year_Processor', 'recommender'], value_vars=list(co2_emissions['2013'].keys()),
                            var_name='Region', value_name='CO2 Emissions (in g)')

    # # Sort the data by total energy consumed
    # melted_data.sort_values(by='Year_Processor', inplace=True, ascending=True)

    # Define hue order and palette explicitly
    hue_order = ['2023 Deep Learning (GPU) \n Algorithm Executed \n on 2023 Hardware',
                 '2013 Traditional (CPU) \n Algorithm Executed \n on 2013 Hardware']

    # Create the plot
    fig, axs = plt.subplots(figsize=(10, 6))

    # Draw the bar plot
    barplot = sns.barplot(data=melted_data, x='CO2 Emissions (in g)', y='Region', hue='Year_Processor', errorbar=None,
                          hue_order=hue_order, palette=[color_leviathan, color_old_pc])

    # Ensure the grid is behind all other plot elements
    axs.set_axisbelow(True)  # This forces the grid to be drawn below plot elements

    # Set the zorder of the grid lines to -1 to ensure they are behind the bars
    axs.grid(True, which='both', axis='x', zorder=-1)

    # Now, manually set the zorder of the bars higher than the grid
    for bar in barplot.patches:
        bar.set_zorder(1)

    plt.title('CO2 Emissions by Region and Algorithm Type')
    plt.xlabel('[estimated] Grams of CO2 Equivalents Emitted (in gCO2e)')
    plt.ylabel('Region')
    plt.legend(title='Algorithm Type/Year')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()


cpu_recommenders = ["ItemKNN (LensKit)", "UserKNN (LensKit)", "ImplicitMF", "Popularity (RecPack)",
                    "ItemKNN (RecPack)", "BPR", "NMF", "SVD"]

traditional_recommenders = ["ItemKNN (LensKit)", "UserKNN (LensKit)", "ImplicitMF", "Popularity (RecPack)",
                            "ItemKNN (RecPack)", "BPR", "NMF", "SVD", "ItemKNN (RecBole)", "Popularity (RecBole)"]

mac_studio_datasets = ['Hetrec-LastFM', 'MovieLens-100K', 'MovieLens-1M', 'MovieLens-Latest-Small',
                       "Amazon2018-Electronics", "Amazon2018-Toys-And-Games",
                       "Amazon2018-Sports-And-Outdoors"]

mac_studio_recommenders = ["PopScore", "ItemItem", "UserUser", "ImplicitMF", "SVD", "NMF", "ItemKNNRP",
                           "ItemKNN", "BPR", "NeuMF", "MultiVAE", "RecVAE"]

# Read the data
old_pc_rating_prediction_data = pd.read_csv('experiment_logs_rating_prediction_old_pc/mapped_logs_overall_old_pc.csv')
old_pc_ranking_prediction_data = pd.read_csv('experiment_logs_ranking_prediction_old_pc/mapped_logs_overall_old_pc.csv')
leviathan_data = pd.read_csv('experiment_logs_leviathan/mapped_logs_overall_leviathan.csv')
macStudio_data = pd.read_csv('experiment_logs_macStudio/mapped_logs_overall_macStudio.csv')
macbook_data = pd.read_csv('experiment_logs_macbook/macbook.csv', sep=';')

# Rename to avoid duplicate labels
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['dataset'] == "MovieLens-1M", 'dataset'] = "MovieLens-1M w/ Ratings"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['dataset'] == "MovieLens-100K", 'dataset'] = "MovieLens-100K w/ Ratings"
leviathan_data.loc[leviathan_data['recommender'] == "ItemItem", 'recommender'] = "ItemKNN (LensKit)"
leviathan_data.loc[leviathan_data['recommender'] == "UserUser", 'recommender'] = "UserKNN (LensKit)"
leviathan_data.loc[leviathan_data['recommender'] == "ItemKNNRP", 'recommender'] = "ItemKNN (RecPack)"
leviathan_data.loc[leviathan_data['recommender'] == "PopScore", 'recommender'] = "Popularity (RecPack)"
leviathan_data.loc[leviathan_data['recommender'] == "Pop", 'recommender'] = "Popularity (RecBole)"
leviathan_data.loc[leviathan_data['recommender'] == "ItemKNN", 'recommender'] = "ItemKNN (RecBole)"

old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "ItemItem", 'recommender'] = "ItemKNN (LensKit)"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "UserUser", 'recommender'] = "UserKNN (LensKit)"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "ItemKNNRP", 'recommender'] = "ItemKNN (RecPack)"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "PopScore", 'recommender'] = "Popularity (RecPack)"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "Pop", 'recommender'] = "Popularity (RecBole)"
old_pc_rating_prediction_data.loc[
    old_pc_rating_prediction_data['recommender'] == "ItemKNN", 'recommender'] = "ItemKNN (RecBole)"

old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "ItemItem", 'recommender'] = "ItemKNN (LensKit)"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "UserUser", 'recommender'] = "UserKNN (LensKit)"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "ItemKNNRP", 'recommender'] = "ItemKNN (RecPack)"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "PopScore", 'recommender'] = "Popularity (RecPack)"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "Pop", 'recommender'] = "Popularity (RecBole)"
old_pc_ranking_prediction_data.loc[
    old_pc_ranking_prediction_data['recommender'] == "ItemKNN", 'recommender'] = "ItemKNN (RecBole)"

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

old_pc_data = pd.concat([old_pc_rating_prediction_data, old_pc_ranking_prediction_data], ignore_index=True)
combined_data = pd.concat([old_pc_data, leviathan_data], ignore_index=True)
combined_data = pd.concat([combined_data, macbook_data], ignore_index=True)
combined_data = pd.concat([combined_data, macStudio_data], ignore_index=True)

combined_data = combined_data[combined_data['recommender'] != 'ProximityBagging']
old_pc_data = old_pc_data[old_pc_data['recommender'] != 'ProximityBagging']
old_pc_rating_prediction_data = old_pc_rating_prediction_data[
    old_pc_rating_prediction_data['recommender'] != 'ProximityBagging']
old_pc_ranking_prediction_data = old_pc_ranking_prediction_data[
    old_pc_ranking_prediction_data['recommender'] != 'ProximityBagging']
leviathan_data = leviathan_data[leviathan_data['recommender'] != 'ProximityBagging']

combined_data = combined_data[combined_data['recommender'] != 'BPR']
old_pc_data = old_pc_data[old_pc_data['recommender'] != 'BPR']
old_pc_rating_prediction_data = old_pc_rating_prediction_data[
    old_pc_rating_prediction_data['recommender'] != 'BPR']
old_pc_ranking_prediction_data = old_pc_ranking_prediction_data[
    old_pc_ranking_prediction_data['recommender'] != 'BPR']

leviathan_data["Algorithm Type"] = "Deep Learning"
leviathan_data.loc[leviathan_data['recommender'].isin(traditional_recommenders), 'Algorithm Type'] = 'Traditional Model'

leviathan_data['total_energy_consumed (total energy consumed, in Wh)'] = leviathan_data[
                                                                             'total_energy_consumed (total energy consumed, in Wh)'] / 1000
old_pc_data['total_energy_consumed (total energy consumed, in Wh)'] = old_pc_data[
                                                                          'total_energy_consumed (total energy consumed, in Wh)'] / 1000
macbook_data['total_energy_consumed (total energy consumed, in Wh)'] = macbook_data[
                                                                           'total_energy_consumed (total energy consumed, in Wh)'] / 1000
macStudio_data['total_energy_consumed (total energy consumed, in Wh)'] = macStudio_data[
                                                                             'total_energy_consumed (total energy consumed, in Wh)'] / 1000
combined_data['total_energy_consumed (total energy consumed, in Wh)'] = combined_data[
                                                                            'total_energy_consumed (total energy consumed, in Wh)'] / 1000

figure1(leviathan_data, 'figures/figure1.svg')
figure2(combined_data, 'figures/figure2.svg')
figure3(leviathan_data, traditional_models=traditional_recommenders, gpu_models=cpu_recommenders,
        save_path='figures/figure3.svg')
figure4(old_pc_data, 'figures/figure4.svg')
figure5(combined_data, 'figures/figure5.svg')
