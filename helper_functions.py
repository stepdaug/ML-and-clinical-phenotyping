import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, t, norm

def mean_confidence_interval_and_p_value(data1, data2, confidence=0.95): # calculate mean, confidence interval, and p-value for difference
    # Calculate the mean, confidence interval, and p-value for the difference between two datasets.
    n1 = len(data1)
    n2 = len(data2)
    m1 = np.mean(data1)
    m2 = np.mean(data2)
    std_err1 = np.std(data1, ddof=1) / np.sqrt(n1)
    std_err2 = np.std(data2, ddof=1) / np.sqrt(n2)
    margin_of_error1 = std_err1 * t.ppf((1 + confidence) / 2., n1 - 1)
    margin_of_error2 = std_err2 * t.ppf((1 + confidence) / 2., n2 - 1)
    lower_ci1 = m1 - margin_of_error1
    upper_ci1 = m1 + margin_of_error1
    lower_ci2 = m2 - margin_of_error2
    upper_ci2 = m2 + margin_of_error2
    p_value = ttest_ind(data1, data2)[1]
    mean_diff = m1 - m2
    lower_ci_diff = mean_diff - (np.sqrt(std_err1**2 + std_err2**2) * t.ppf((1 + confidence) / 2., n1 + n2 - 2))
    upper_ci_diff = mean_diff + (np.sqrt(std_err1**2 + std_err2**2) * t.ppf((1 + confidence) / 2., n1 + n2 - 2))
    return m1, lower_ci1, upper_ci1, m2, lower_ci2, upper_ci2, mean_diff, lower_ci_diff, upper_ci_diff, p_value

def replace_words_present(column_name):
    if column_name == "Sex":
        return "Male"
    elif column_name == "V1":
        return "Present"
    elif column_name == "Z":
        return "Present"
    else:
        parts = column_name.split()
        if parts[0] == "Symptom":
            return "Has symp " + parts[1]
        elif parts[0] == "Variable":
            # return "Variable " + parts[1] + " present"
            return "Present"
        else:
            return column_name
    
def replace_words_absent(column_name):
    if column_name == "Sex":
        return "Female"
    elif column_name == "V1":
        return "Absent"
    elif column_name == "Z":
        return "Absent"
    else:
        parts = column_name.split()
        if parts[0] == "Demographic":
            return "Not Dem " + parts[1]
        elif parts[0] == "Symptom":
            return "No symp " + parts[1]
        elif parts[0] == "Variable":
            # return "Variable " + parts[1] + " absent"
            return "Absent"
        else:
            return column_name

def add_letter(ax, letter, f_size): # Function to add bold letters to the top left corner of each panel
    ax.text(0.0, 1.07, letter, transform=ax.transAxes, fontsize=f_size, fontweight='bold', va='top', ha='right')

def binomial_ci(accuracy, n, confidence=0.95):
    z = norm.ppf((1 + confidence) / 2)
    margin_of_error = z * np.sqrt(accuracy * (1 - accuracy) / n)
    lower_bound = accuracy - margin_of_error
    upper_bound = accuracy + margin_of_error
    return lower_bound, upper_bound

def accuracy_and_ci(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    total_n = tp + fn + fp + tn
    accuracy = (tp + tn)/(tp + fn + fp + tn)
    lower_ci, upper_ci = binomial_ci(accuracy, total_n)
    print(accuracy, lower_ci, upper_ci)

def plot_confusion_matrix_with_metrics(ax, cm, title, font_size): # plot detailed confusion matrix with metrics
    # tp, fn, fp, tn = cm # original
    tn, fp, fn, tp = cm
    total_n = tp + fn + fp + tn
    accuracy = (tp + tn)/(tp + fn + fp + tn) *100
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    if "trial outcome" in title.lower():
        cm_table = [
            ["", "Outcome\nResponsive", "Outcome\nNot Responsive", ""],
            ["Predicted\nResponsive", tp, fp, f"PPV\n{round(ppv, 2):.2f}"],
            ["Predicted\nNot\nResponsive", fn, tn, f"NPV\n{round(npv, 2):.2f}"],            
            ["", f"Sensitivity\n{round(sensitivity, 2):.2f}", f"Specificity\n{round(specificity, 2):.2f}", f"{round(accuracy,1):.1f}%\naccurate"],
        ]
    elif "ground truth" in title.lower():
        cm_table = [
            ["", "Truth\nResponsive", "Truth\nNot Responsive", ""],
            ["Predicted\nResponsive", tp, fp, f"PPV\n{round(ppv, 2):.2f}"],
            ["Predicted\nNot\nResponsive", fn, tn, f"NPV\n{round(npv, 2):.2f}"],
            ["", f"Sensitivity\n{round(sensitivity, 2):.2f}", f"Specificity\n{round(specificity, 2):.2f}", f"{round(accuracy,1):.1f}%\naccurate"],
        ]

    pos_col = "g" # forestgreen
    neg_col = "r" # indianred
    neut_col = "lemonchiffon"
    title_col = "silver"
    cell_colors = [
        ["w", title_col, title_col, "w"],
        [title_col, pos_col, neg_col, neut_col],
        [title_col, pos_col, neg_col, neut_col],
        ["w", neut_col, neut_col, neut_col],
    ]
    cell_alphas = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.5, 1.0],
        [1.0, 0.5, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    ax.axis('off')
    column_widths = [0.22, 0.30, 0.30, 0.18]
    table = ax.table(cellText=cm_table, cellLoc='center', loc='center', cellColours=cell_colors, colWidths=column_widths)
    
    # Set the alpha values for each cell
    for i, row in enumerate(cm_table):
        for j, cell in enumerate(row):
            table[(i, j)].set_alpha(cell_alphas[i][j])

    table.auto_set_font_size(False)
    table.set_fontsize(font_size-(font_size/11))
    # table.set_fontsize(font_size)
    # Set heights of cells in each row
    row_heights = [0.14, 0.30, 0.30, 0.14] # height of row 1/2/3/4
    cellDict = table.get_celld()
    for i in range(0,len(cm_table[0])): # for each column
        for j in range(0,len(cm_table)):
            cellDict[(j,i)].set_height(row_heights[j])
            
    # Make the true positive/true negative/false positive/false negative values bold too
    for i in range(1,3):
        for j in range(1,3):
            table[i, j].get_text().set_fontweight('bold')  

    ax.set_title(title, fontsize=font_size*1.1)
