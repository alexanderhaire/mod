"""
Generate Strategy Allocation Report (PDF)
=========================================

Generates a professional PDF report on the optimal portfolio allocation between
ERP Regime, AlphaMax, and Compounder strategies.

RUN: python generate_allocation_report.py
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import datetime

# =============================================================================
# DATA
# =============================================================================

REPORT_FILENAME = "Strategy_Allocation_Report.pdf"
DATE_STR = datetime.datetime.now().strftime("%Y-%m-%d")

SUMMARY = {
    "title": "Strategy Allocation Analysis",
    "subject": "Portfolio Optimization (ERP vs AlphaMax vs Compounder)",
    "thesis": [
        "We tested whether combining the three validated strategies creates a",
        "better risk-adjusted portfolio than any single strategy alone.",
        "",
        "VERDICT: MARGINAL BENEFIT (+0.03 Sharpe)",
        "The strategies are highly correlated (>0.60), limiting diversification benefits.",
        "The optimal portfolio concentrates 60% in AlphaMax and 40% in ERP Regime,",
        "dropping the Compounder strategy entirely.",
        "",
        "RECOMMENDATION:",
        "Focus on AlphaMax (The Bedrock) or a simplistic 50/50 split with ERP.",
        "Do not over-complicate execution with a 3-strategy silo."
    ],
    "matrix": [
        ["Strategy", "Sharpe", "CAGR", "Vol", "Allocation"],
        ["AlphaMax", "1.17", "20.4%", "17.4%", "60%"],
        ["ERP Regime", "1.02", "11.4%", "11.2%", "40%"],
        ["Compounder", "0.89", "8.7%", "9.7%", "0%"],
        ["Portfolio", "1.21", "16.8%", "13.9%", "100%"]
    ],
    "correlations": [
        ["", "ERP", "AlphaMax", "Comp"],
        ["ERP", "1.00", "0.70", "0.63"],
        ["AlphaMax", "0.70", "1.00", "0.80"],
        ["Comp", "0.63", "0.80", "1.00"]
    ],
    "conclusion": [
        "1. High Correlation: A correlation of 0.80 between AlphaMax and Compounder",
        "   means they are effectively trading the same risk factors.",
        "2. Alpha Dominance: AlphaMax captures the same momentum/macro edge as",
        "   Compounder but with better execution (Sharpe 1.17 vs 0.89).",
        "3. ERP Uncorrelated-ish: ERP has the lowest correlation (0.63-0.70),",
        "   providing the only meaningful diversification benefit."
    ]
}

def draw_header(ax, title, subject):
    ax.text(0.05, 0.95, title, fontsize=24, fontweight='bold', color='#1a237e')
    ax.text(0.05, 0.90, "Comprehensive Startegy Intelligence Report", fontsize=10, color='gray')
    ax.text(0.05, 0.87, f"Subject: {subject}", fontsize=10, fontstyle='italic')
    ax.plot([0.05, 0.95], [0.85, 0.85], color='black', lw=2)
    ax.axis('off')

def draw_section_header(ax, text, y):
    ax.text(0.05, y, text, fontsize=14, fontweight='bold', color='#283593')
    ax.plot([0.05, 0.95], [y-0.02, y-0.02], color='#283593', lw=1)

def draw_table(ax, data, y_start, col_widths):
    x = 0.05
    for i, header in enumerate(data[0]):
        ax.text(x, y_start, header, fontsize=10, fontweight='bold', bbox=dict(facecolor='#e8eaf6', edgecolor='none', pad=5))
        x += col_widths[i]
    
    y = y_start - 0.05
    for row in data[1:]:
        x = 0.05
        for i, cell in enumerate(row):
            weight = 'bold' if i == 0 else 'normal'
            color = 'black'
            # Highlight Portfolio row
            if row[0] == "Portfolio": weight = 'bold'; color = '#1a237e'
            
            ax.text(x, y, cell, fontsize=10, fontweight=weight, color=color)
            x += col_widths[i]
        
        ax.plot([0.05, 0.95], [y-0.015, y-0.015], color='#eeeeee', lw=0.5)
        y -= 0.04
    return y

def create_report():
    print(f"Generating {REPORT_FILENAME}...")
    with PdfPages(REPORT_FILENAME) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0, 0, 1, 1])
        draw_header(ax, SUMMARY['title'], SUMMARY['subject'])
        
        # 1. Executive Summary
        y = 0.80
        draw_section_header(ax, "1. Executive Summary & Verdict", y)
        y -= 0.05
        for line in SUMMARY['thesis']:
            ax.text(0.07, y, line, fontsize=10)
            y -= 0.025
            
        # 2. Performance Matrix
        y -= 0.05
        draw_section_header(ax, "2. Optimal Allocation (Max Sharpe)", y)
        y = draw_table(ax, SUMMARY['matrix'], y-0.05, col_widths=[0.25, 0.15, 0.15, 0.15, 0.2])
        
        # 3. Correlation Matrix
        y -= 0.05
        draw_section_header(ax, "3. Correlation Matrix (Why Diversification Failed)", y)
        y = draw_table(ax, SUMMARY['correlations'], y-0.05, col_widths=[0.25, 0.2, 0.2, 0.2])
        
        # 4. Conclusion
        y -= 0.05
        draw_section_header(ax, "4. Strategic Conclusion", y)
        y -= 0.05
        for line in SUMMARY['conclusion']:
             ax.text(0.07, y, line, fontsize=10, wrap=True)
             y -= 0.03
             
        pdf.savefig(fig)
        plt.close()
    print("Done!")

if __name__ == "__main__":
    create_report()
