import pandas as pd
import matplotlib.pyplot as plt

def generate_chart():
    # Load data
    df = pd.read_csv('cost_changes_sep30_vs_today.csv')
    
    # Take top 15 by absolute difference
    top_df = df.sort_values('AbsDiff', ascending=False).head(15).sort_values('AbsDiff', ascending=True)
    
    # Prepare Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bar chart of differences
    # Green for increase (positive diff), Red for decrease (negative diff)
    colors = ['green' if x > 0 else 'red' for x in top_df['Diff']]
    bars = ax.barh(top_df['ITEMNMBR'], top_df['Diff'], color=colors)
    
    # Titles and Labels
    ax.set_title('Top 15 Cost Changes ($) - Sep 30 vs Today', fontsize=16)
    ax.set_xlabel('Cost Difference ($)', fontsize=12)
    ax.set_ylabel('Item', fontsize=12)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x = width + (width * 0.05) if width > 0 else width + (width * 0.05) -5 # Adjust position
        # For negative bars, put label to the left
        ha = 'left' if width > 0 else 'right'
        
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:+.2f}', va='center', ha=ha, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cost_changes_summary.png', dpi=100)
    print("Chart saved to cost_changes_summary.png")

if __name__ == "__main__":
    generate_chart()
