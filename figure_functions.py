import pandas as pd
import numpy as np
import forestplot as fp # pip install forestplot; https://github.com/LSYS/forestplot?tab=readme-ov-file
import shap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
from helper_functions import *

def figure_2(subgroup_columns,outcome_table):
    fig_width = 1.2*len(subgroup_columns) # 0.4
    fig_height = 1.5*len(subgroup_columns) # 1.0
    fnt_sz_xlab = 22 # 14
    fnt_sz_ylab = 13 # 12
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(fig_width,fig_height))  # Adjust figsize as needed
    fp.forestplot(outcome_table,  # the dataframe with results data
                  estimate="Mean Difference",  # col containing estimated effect size 
                  ll="Difference Lower CI", hl="Difference Upper CI",  # columns containing conf. int. lower and higher limits
                  capitalize="capitalize",  # Capitalize labels
                  # varlabel="Group",  # column containing variable label
                  varlabel="Label",  # column containing variable label
                  groupvar="Group_variable",  # Add variable groupings 
                  sort=False,  # sort in ascending order (sorts within group if group is specified)               
                  color_alt_rows=True,  # Gray alternate rows
                  ylabel="Treatment 1",  # ylabel to print
                   **{"ylabel1_size": 16},  # control size of printed ylabel
                  xlabel="Change from baseline",  # x-label title
                  **{"xlabel1_size": 15},  # control size of printed x label
                  xticks=[-2, 0, 2, 4, 6, 8], # for if I need to standardise x axis for comparison between plots
                  # Additional kwargs for customizations
                   **{"marker": "D", # diamond marker
                       "markersize": 30,
                       "offset": 0.35,# override default vertical offset between models (0.0 to 1.0)
                       "xlinestyle": (0, (10, 5)),  # long dash for x-reference line
                       "xtick_size": fnt_sz_xlab,  # adjust x-ticker fontsize
                   },
                  decimal_precision = 2, # how many d.p.
                  ax=axes
                  )
    # Adjust the y-label positions
    axes.set_ylabel("", fontsize=0)  # Adjust the fontsize
    axes.set_xlabel("Mean (95% CI) difference treatment vs placebo", fontsize=fnt_sz_xlab, fontweight='normal')  # Adjust the fontsize
    axes.xaxis.set_label_coords(0.27, -0.05)
    
    # Set the font for the y-axis label manually
    for label in axes.get_yticklabels():
        label.set_fontsize(fnt_sz_ylab)  # Set the desired font size
    
    plt.tight_layout()  # Minimize overlap of subplots
    plt.show()

def figure_3(conf_matrix_xgb_trial,conf_matrix_xgb_truth):
    font_sz = 115
    outcome = ['trial','true'] # which outcome metric (trial data or ground truth) to plot confusion matrices/ROC curves for
    fig_ML, axes_ML = plt.subplots(1, 2, figsize=(80, 40))
    for outc in outcome:
        if outc == 'trial':
            # cm_rct = conf_matrix_rct_trial
            # cm_lr = conf_matrix_lr_trial
            cm_xgb = conf_matrix_xgb_trial
            ax_cm_rct = axes_ML[0]
            ax_cm_xgb = axes_ML[0]
            fig_letter_addder = 0
            title_addition = 'Trial outcome'
        elif outc == 'true':
            # cm_rct = conf_matrix_rct_truth
            # cm_lr = conf_matrix_lr_truth
            cm_xgb = conf_matrix_xgb_truth
            ax_cm_log_reg = axes_ML[1]
            ax_cm_rct = axes_ML[1]
            ax_cm_xgb = axes_ML[1]
            fig_letter_addder = 1
            title_addition = 'Ground truth'
    
        # Plot confusion matrix for XGBoost
        plot_confusion_matrix_with_metrics(ax_cm_xgb, cm_xgb.ravel(), f'XGB predictions vs {title_addition}',font_sz)
        add_letter(ax_cm_xgb, chr(65 + fig_letter_addder), font_sz*1.5)
    
    plt.figure(fig_ML)
    plt.tight_layout()
    plt.show()
 
def figure_4(shap_values,X_trial_scaled,feature_names):
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(18, 12))
    cmap = cm.get_cmap("Blues")  # You can replace 'viridis' with any valid colormap
    shap.summary_plot(shap_values, X_trial_scaled, feature_names=feature_names, plot_type="dot", color_bar=False,show=False)
    ax = plt.gca()
    for collection in ax.collections:
        collection.set_cmap(cmap)
        collection.set_edgecolor("black")  # Set black outline
        collection.set_linewidth(0.2) # thickness of outline
        collection.set_sizes([20])  # Set marker size
    
    cbar = plt.colorbar(ax.collections[0], ax=ax)  # Add a colorbar for reference
    cbar.set_ticks([])  # Remove the ticks
    cbar.ax.text(0.5, -0.03, "Low", ha='center', va='top', fontsize=12)   # Add 'low' at the bottom
    cbar.ax.text(0.5, 1.03, "High", ha='center', va='bottom', fontsize=12) # Add 'high' at the top
    
    plt.rcParams['font.family'] = 'Arial'
    plt.xticks(fontname='Arial')
    plt.yticks(fontname='Arial')
    plt.xlabel('SHAP value (impact on model output)', fontsize=14)
    
    plt.tight_layout()
    plt.show()

def figure_5(y_trial,X_trial_before_scaled,shap_values,factors_to_exclude):
    # FIGURE 5A
    # X has some very high shap values conditions - look at those, it's high values - this is immediarely below - scatter of x values vs shap values, with histogram of positive/negative predictions in each bin
    # Then exclude those X>90/95 values and look at Y, Z dependence plot interaction - see the 50-90 range important, 
    colors = [(0, "r"), (0.5, "y"), (1, "g")]  # Color transitions from red to green
    cmap = LinearSegmentedColormap.from_list("my_custom_colormap", colors)
    X_col = 4; Y_col = 5; Z_col = 6
    # Set plot properties
    plot_labl_fontsize = 35 # the A/B/C subplot labels
    title_fontsize = 32
    label_fontsize = 28
    tick_fontsize = 24
    bar_alpha = 0.1; scatter_alpha=1.0
    scatter_outline_thickness = 0.6
    scatter_plot_sz = 60
    y1_tick_vals = [-4,-2,0,2,4]
    y2_tick_vals = [0,0.5,1]
    # Interrogate Z
    feature_index = X_col  # Example feature index; 4 = X, 5 = Y, 6 = Z
    predictions = y_trial
    fig, ax1 = plt.subplots(1, 3, figsize=(21, 7))
    
    # Scatter plot with color indicating predictions
    scatter = ax1[0].scatter(
        X_trial_before_scaled[:, feature_index],
        shap_values[:, feature_index],  # SHAP values for y-axis
        c=predictions,  # Color by predictions
        cmap=cmap,
        edgecolor='k',  # Optional: add edge color for better visibility
        linewidth=scatter_outline_thickness, #thickness of black outline for the plots
        s=scatter_plot_sz,
        alpha=scatter_alpha  # Optional: adjust transparency
    )
    
    # Set labels and title
    ax1[0].set_xlabel('Value of X', fontsize=label_fontsize)
    ax1[0].set_ylabel('SHAP value', fontsize=label_fontsize)
    ax1[0].set_title('SHAP Scatter Plot for X', fontsize=title_fontsize)
    ax1[0].text(-0.05, 1.1, 'A', transform=ax1[0].transAxes, fontsize=plot_labl_fontsize, fontweight='bold', va='top', ha='right')
    ax1[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1[0].set_yticks(y1_tick_vals)
    
    # Define bins for histogram
    bin_edges = np.arange(50, 101, 5)  # Example intervals from -15 to 15 with step of 5
    hist_counts, _ = np.histogram(shap_values[:, feature_index], bins=bin_edges) # Count SHAP values within each bin
    hist_counts_pos, _ = np.histogram(X_trial_before_scaled[:, feature_index][predictions == 1], bins=bin_edges) # Count SHAP values within each bin
    hist_counts_neg, _ = np.histogram(X_trial_before_scaled[:, feature_index][predictions == 0], bins=bin_edges) # Count SHAP values within each bin
    
    # Bin centers for plotting
    ax2 = ax1[0].twinx()  # Create a twin y-axis that shares the same x-axis
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ax2.bar(bin_centers, hist_counts_pos / (hist_counts_pos + hist_counts_neg), 
            width=5, color='g', edgecolor='black', alpha=bar_alpha, label='predict responsive')
    
    ax2.bar(bin_centers, hist_counts_neg / (hist_counts_pos + hist_counts_neg),
            width=5, bottom=hist_counts_pos / (hist_counts_pos + hist_counts_neg),
            color='r', edgecolor='black', alpha=bar_alpha, label='predict not responsive')
    
    ax1[0].set_ylim(-4, 4)
    ax2.set_ylim(0, 1)
    ax2.set_yticks(y2_tick_vals)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            
    # FIGURE 5B
    # Filter out those with X above 90 which we already know X is important for to consider their relationships with other variables
    filtered_X = X_trial_before_scaled[X_trial_before_scaled[:, X_col] < 90]
    filtered_shap = shap_values[X_trial_before_scaled[:, X_col] < 90]
    filtered_predictions = predictions[X_trial_before_scaled[:, X_col] < 90]
    feature_index = Y_col  # Example feature index; 4 = X, 5 = Y, 6 = Z
    
    # Scatter plot with color indicating predictions
    scatter = ax1[1].scatter(
        filtered_X[:, feature_index],
        filtered_shap[:, feature_index],  # SHAP values for y-axis
        c=filtered_predictions,  # Color by predictions
        cmap=cmap,
        edgecolor='k',  # Optional: add edge color for better visibility
        linewidth=scatter_outline_thickness, #thickness of black outline for the plots
        s=scatter_plot_sz,
        alpha=scatter_alpha  # Optional: adjust transparency
    )
    
    # Set labels and title
    ax1[1].set_xlabel('Value of Y', fontsize=label_fontsize)
    ax1[1].set_title('SHAP Scatter Plot for Y', fontsize=title_fontsize)
    ax1[1].text(-0.05, 1.1, 'B', transform=ax1[1].transAxes, fontsize=plot_labl_fontsize, fontweight='bold', va='top', ha='right')
    ax1[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1[1].set_yticks(y1_tick_vals)
    
    # Define bins for histogram
    bin_edges = np.arange(40, 101, 5)  # Example intervals from -15 to 15 with step of 5
    hist_counts, _ = np.histogram(filtered_shap[:, feature_index], bins=bin_edges) # Count SHAP values within each bin
    hist_counts_pos, _ = np.histogram(filtered_X[:, feature_index][filtered_predictions == 1], bins=bin_edges) # Count SHAP values within each bin
    hist_counts_neg, _ = np.histogram(filtered_X[:, feature_index][filtered_predictions == 0], bins=bin_edges) # Count SHAP values within each bin
    
    # Bin centers for plotting
    ax2 = ax1[1].twinx()  # Create a twin y-axis that shares the same x-axis
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ax2.bar(bin_centers, hist_counts_pos / (hist_counts_pos + hist_counts_neg), 
            width=5, color='g', edgecolor='black', alpha=bar_alpha, label='predict responsive')
    
    ax2.bar(bin_centers, hist_counts_neg / (hist_counts_pos + hist_counts_neg),
            width=5, bottom=hist_counts_pos / (hist_counts_pos + hist_counts_neg),
            color='r', edgecolor='black', alpha=bar_alpha, label='predict not responsive')
    
    ax1[1].set_ylim(-4, 4)
    ax2.set_ylim(0, 1)
    ax2.set_yticks(y2_tick_vals)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # plt.show()
    
    # FIGURE 5C
    # Filter out those with Y <50 and >90 which we already know Y is important for to consider their relationships with other variables
    filtered_shap = filtered_shap[(filtered_X[:, Y_col] < 90) & (filtered_X[:, Y_col] > 50)]
    filtered_predictions = filtered_predictions[(filtered_X[:, Y_col] < 90) & (filtered_X[:, Y_col] > 50)]
    filtered_X = filtered_X[(filtered_X[:, Y_col] < 90) & (filtered_X[:, Y_col] > 50)]
    
    if "Z" in factors_to_exclude:
        pass # don't try to plot Z if its been excluded
    else:
        feature_index = Z_col  # Example feature index; 4 = X, 5 = Y, 6 = Z
        
        # Scatter plot with color indicating predictions
        scatter = ax1[2].scatter(
            filtered_X[:, feature_index],
            filtered_shap[:, feature_index],  # SHAP values for y-axis
            c=filtered_predictions,  # Color by predictions
            # cmap='coolwarm',  # Color map (red for positive, blue for negative)
            cmap=cmap,
            edgecolor='k',  # Optional: add edge color for better visibility
            linewidth=scatter_outline_thickness, #thickness of black outline for the plots
            s=scatter_plot_sz,
            alpha=scatter_alpha  # Optional: adjust transparency
        )
        
        # Set labels and title
        ax1[2].set_xlabel('Value of Z', fontsize=label_fontsize)
        ax1[2].set_title('SHAP Scatter Plot for Z', fontsize=title_fontsize)
        ax1[2].text(-0.05, 1.1, 'C', transform=ax1[2].transAxes, fontsize=plot_labl_fontsize, fontweight='bold', va='top', ha='right')
        ax1[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax1[2].set_yticks(y1_tick_vals)
        
        # Define bins for histogram
        bin_edges = np.arange(-0.5, 1.6, 1) # Example intervals from -15 to 15 with step of 5
        hist_counts, _ = np.histogram(filtered_shap[:, feature_index], bins=bin_edges) # Count SHAP values within each bin
        hist_counts_pos, _ = np.histogram(filtered_X[:, feature_index][filtered_predictions == 1], bins=bin_edges) # Count SHAP values within each bin
        hist_counts_neg, _ = np.histogram(filtered_X[:, feature_index][filtered_predictions == 0], bins=bin_edges) # Count SHAP values within each bin
        
        # Bin centers for plotting
        ax2 = ax1[2].twinx()  # Create a twin y-axis that shares the same x-axis
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax2.bar(bin_centers, hist_counts_pos / (hist_counts_pos + hist_counts_neg), 
                width=0.5, color='g', edgecolor='black', alpha=bar_alpha, label='predict responsive')
        
        ax2.bar(bin_centers, hist_counts_neg / (hist_counts_pos + hist_counts_neg),
                width=0.5, bottom=hist_counts_pos / (hist_counts_pos + hist_counts_neg),
                color='r', edgecolor='black', alpha=bar_alpha, label='predict not responsive')
        
        ax1[2].set_ylim(-4, 4)
        ax1[2].set_xlim([-0.5, 1.5])
        ax1[2].set_xticks([0, 1])
        ax2.set_ylabel('Prediction proportions', fontsize=label_fontsize)
        ax2.set_ylim(0, 1)
        ax2.set_yticks(y2_tick_vals)
        ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.tight_layout()
    plt.show()