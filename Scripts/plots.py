import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results,model):
    models = [result['model'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    precisions = [result['precision'] for result in results]
    recalls = [result['recall'] for result in results]
    f1_scores = [result['f1-score'] for result in results]

    x = np.arange(len(models))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))

    # Use softer colors for the bars
    ax.bar(x - width * 2, accuracies, width, label='Accuracy', color='#66c2a5')
    ax.bar(x - width, precisions, width, label='Precision', color='#fc8d62')
    ax.bar(x, recalls, width, label='Recall', color='#8da0cb')
    ax.bar(x + width, f1_scores, width, label='F1-score', color='#e78ac3')

    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title(f'{model} Classification Metrics ', fontsize=16)
    
    # Set x-tick labels with rotation
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)

    # Add gridlines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax.legend(fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_results_table(results):
    # Prepare data for the table
    models = [result['model'] for result in results]
    times = [result['time'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    precisions = [result['precision'] for result in results]
    recalls = [result['recall'] for result in results]
    f1_scores = [result['f1-score'] for result in results]

    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hide the axes
    ax.axis('off')

    # Create a table
    table_data = [['Model', 'Time', 'Accuracy', 'Precision', 'Recall', 'F1-score']]
    
    # Append data for each model
    for i in range(len(models)):
        table_data.append([models[i], f"{times[i]:.2f}", f"{accuracies[i]:.2f}", 
                           f"{precisions[i]:.2f}", f"{recalls[i]:.2f}", f"{f1_scores[i]:.2f}"])

    # Create the table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colLabels=None)

    # Customize the table appearance
    table.auto_set_font_size(False)  # Disable auto font size
    table.set_fontsize(12)  # Set a consistent font size
    
    # Set column widths and row heights
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_fontsize(14)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#D9EAD3')  # Header color
            cell.set_edgecolor('#A3A3A3')  # Header border color
            cell.set_height(0.15)  # Header row height
        else:  # Data rows
            cell.set_facecolor('#F9F9F9')  # Row color
            cell.set_edgecolor('#A3A3A3')  # Cell border color
            cell.set_height(0.10)  # Row height
            cell.set_fontsize(12)  # Font size for data rows
            cell.set_text_props(weight='normal')  # Regular font weight
            cell.set_facecolor('#F5F5F5')  # Slightly darker for distinction

    # Optional: Add a title to the plot
    #plt.title('Model Performance Metrics', fontsize=16, weight='bold', color='black')

    plt.show()
