"""
Generate Setup Guide PDF for Ultimate Combined Strategy
========================================================

Creates a professional PDF with step-by-step setup instructions.

RUN: python generate_setup_guide.py
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os

def create_setup_guide():
    output_path = r"c:\Users\alexh\Downloads\mod\Ultimate_Strategy_Setup_Guide.pdf"
    
    doc = SimpleDocTemplate(output_path, pagesize=letter, 
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=24, 
                                  spaceAfter=30, textColor=colors.HexColor('#1a5f7a'))
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=16, 
                               spaceBefore=20, spaceAfter=10, textColor=colors.HexColor('#1a5f7a'))
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, 
                               spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#2d3436'))
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=11, 
                                 spaceAfter=8, leading=14)
    code_style = ParagraphStyle('Code', parent=styles['Code'], fontSize=10, 
                                 backColor=colors.HexColor('#f5f5f5'), 
                                 leftIndent=20, rightIndent=20, spaceBefore=5, spaceAfter=10)
    bullet_style = ParagraphStyle('Bullet', parent=body_style, leftIndent=20, bulletIndent=10)
    important_style = ParagraphStyle('Important', parent=body_style, 
                                      backColor=colors.HexColor('#fff3cd'), 
                                      leftIndent=10, rightIndent=10, spaceBefore=10, spaceAfter=10)
    
    story = []
    
    # TITLE PAGE
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("ULTIMATE COMBINED STRATEGY", title_style))
    story.append(Paragraph("Setup & Deployment Guide", ParagraphStyle('Subtitle', parent=styles['Heading2'], 
                                                                        fontSize=18, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Validated Strategy • 5/6 Tests Passed • Sharpe 1.42", 
                           ParagraphStyle('Tagline', parent=body_style, alignment=TA_CENTER, textColor=colors.grey)))
    story.append(Spacer(1, 1*inch))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Sharpe Ratio', '1.42'],
        ['CAGR', '28.6%'],
        ['Max Drawdown', '-27.5%'],
        ['Validation p-value', '0.0002'],
        ['Walk-Forward', '3/3 folds positive'],
    ]
    summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5f7a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(summary_table)
    
    story.append(PageBreak())
    
    # SECTION 1: OVERVIEW
    story.append(Paragraph("1. Strategy Overview", h1_style))
    story.append(Paragraph("""
    The Ultimate Combined Strategy allocates 60% to traditional assets (SPY, TLT, GLD) and 40% to 
    cryptocurrency (BTC plus altcoins). It uses two timing signals:
    """, body_style))
    
    story.append(Paragraph("• <b>VIX Term Structure</b> - For traditional asset timing", bullet_style))
    story.append(Paragraph("• <b>Altseason Detector</b> - For crypto asset selection", bullet_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Allocation table
    story.append(Paragraph("Portfolio Allocation", h2_style))
    alloc_data = [
        ['Sleeve', 'Allocation', 'Signal Used'],
        ['Traditional (SPY/TLT/GLD)', '60%', 'VIX Term Structure'],
        ['Crypto (BTC/Alts)', '40%', 'Altseason + BTC Momentum'],
    ]
    alloc_table = Table(alloc_data, colWidths=[2.2*inch, 1.2*inch, 2*inch])
    alloc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5f7a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(alloc_table)
    
    story.append(PageBreak())
    
    # SECTION 2: PREREQUISITES
    story.append(Paragraph("2. Prerequisites", h1_style))
    
    story.append(Paragraph("Required Software", h2_style))
    story.append(Paragraph("• Python 3.8 or higher", bullet_style))
    story.append(Paragraph("• pip (Python package manager)", bullet_style))
    
    story.append(Paragraph("Required Python Packages", h2_style))
    story.append(Paragraph("pip install numpy pandas yfinance scipy", code_style))
    
    story.append(Paragraph("Required Accounts (for live trading)", h2_style))
    story.append(Paragraph("• Brokerage account (Interactive Brokers, Alpaca, etc.)", bullet_style))
    story.append(Paragraph("• Crypto exchange account (Coinbase, Binance, Kraken)", bullet_style))
    
    # SECTION 3: FILES
    story.append(Paragraph("3. Strategy Files", h1_style))
    
    files_data = [
        ['File', 'Purpose'],
        ['ultimate_strategy.py', 'Main strategy implementation'],
        ['falsify_ultimate_combined.py', 'Validation tests'],
        ['deep_crypto.py', 'Altseason detector'],
        ['crypto_signals.py', 'Crypto signal generation'],
    ]
    files_table = Table(files_data, colWidths=[2.5*inch, 3.5*inch])
    files_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5f7a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(files_table)
    
    story.append(PageBreak())
    
    # SECTION 4: SIGNAL CALCULATION
    story.append(Paragraph("4. Signal Calculation", h1_style))
    
    story.append(Paragraph("VIX Term Structure Signal", h2_style))
    story.append(Paragraph("""
    The VIX signal compares spot VIX to VIX 3-month futures (VIX3M):
    """, body_style))
    
    vix_signal = """
ratio = VIX / VIX3M

if ratio < 0.90:
    signal = "CONTANGO" → Risk-On (overweight SPY)
elif ratio > 1.05:
    signal = "BACKWARDATION" → Risk-Off (overweight TLT)
else:
    signal = "NEUTRAL"
"""
    story.append(Paragraph(vix_signal.replace('\n', '<br/>'), code_style))
    
    story.append(Paragraph("Altseason Signal", h2_style))
    story.append(Paragraph("""
    The Altseason signal compares altcoin momentum to Bitcoin momentum:
    """, body_style))
    
    alt_signal = """
btc_momentum = BTC_price.pct_change(14)
alt_momentum = mean([alt.pct_change(14) for alt in altcoins])

if alt_momentum > btc_momentum * 1.2:
    signal = "ALTSEASON" → Buy top 3 altcoins
else:
    signal = "BTC_DOMINANCE" → Hold BTC
"""
    story.append(Paragraph(alt_signal.replace('\n', '<br/>'), code_style))
    
    story.append(PageBreak())
    
    # SECTION 5: ALLOCATION RULES
    story.append(Paragraph("5. Allocation Rules", h1_style))
    
    story.append(Paragraph("Traditional Sleeve (60%)", h2_style))
    
    trad_data = [
        ['VIX Signal', 'SPY', 'TLT', 'GLD'],
        ['Contango (bullish)', '45%', '10%', '5%'],
        ['Neutral', '30%', '22%', '8%'],
        ['Backwardation (bearish)', '15%', '35%', '10%'],
    ]
    trad_table = Table(trad_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    trad_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(trad_table)
    
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Crypto Sleeve (40%)", h2_style))
    
    crypto_data = [
        ['Crypto Signal', 'BTC', 'Top 3 Alts', 'ETH'],
        ['Strong Altseason', '10%', '30% (10% each)', '-'],
        ['Moderate Altseason', '25%', '-', '15%'],
        ['Neutral', '20%', '-', '-'],
        ['Bearish', '5%', '-', '-'],
    ]
    crypto_table = Table(crypto_data, colWidths=[1.6*inch, 1*inch, 1.6*inch, 1*inch])
    crypto_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f7931a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(crypto_table)
    
    story.append(PageBreak())
    
    # SECTION 6: STEP BY STEP
    story.append(Paragraph("6. Step-by-Step Setup", h1_style))
    
    story.append(Paragraph("Step 1: Install Dependencies", h2_style))
    story.append(Paragraph("pip install numpy pandas yfinance scipy", code_style))
    
    story.append(Paragraph("Step 2: Run the Strategy", h2_style))
    story.append(Paragraph("cd c:\\Users\\alexh\\Downloads\\mod<br/>python ultimate_strategy.py", code_style))
    
    story.append(Paragraph("Step 3: Check Today's Signals", h2_style))
    story.append(Paragraph("""
    The script will output the current signals and recommended allocations. Look for:
    """, body_style))
    story.append(Paragraph("• VIX Signal: Contango/Neutral/Backwardation", bullet_style))
    story.append(Paragraph("• Altseason Signal: Active/Inactive", bullet_style))
    story.append(Paragraph("• Recommended weights for each asset", bullet_style))
    
    story.append(Paragraph("Step 4: Execute Trades", h2_style))
    story.append(Paragraph("""
    Based on the output, execute trades to match the recommended allocation:
    """, body_style))
    story.append(Paragraph("• Use limit orders to minimize slippage", bullet_style))
    story.append(Paragraph("• For crypto, place orders on your exchange", bullet_style))
    story.append(Paragraph("• For traditional ETFs, place orders in your brokerage", bullet_style))
    
    story.append(Paragraph("Step 5: Daily Monitoring", h2_style))
    story.append(Paragraph("""
    Run the strategy daily and rebalance when signals change or drift exceeds 5%.
    """, body_style))
    
    story.append(PageBreak())
    
    # SECTION 7: RISK MANAGEMENT
    story.append(Paragraph("7. Risk Management", h1_style))
    
    story.append(Paragraph("""
    ⚠️ IMPORTANT: This strategy has significant risk. Follow these rules:
    """, important_style))
    
    story.append(Paragraph("Position Limits", h2_style))
    risk_data = [
        ['Control', 'Threshold', 'Action'],
        ['Max crypto allocation', '40%', 'Never exceed'],
        ['Max single alt', '12%', 'Trim if exceeded'],
        ['Portfolio drawdown', '-15%', 'Reduce crypto to 20%'],
        ['BTC volatility', '>100% ann.', 'Reduce crypto to 20%'],
        ['VIX spike', '>35', 'Force defensive mode'],
    ]
    risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 2*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(risk_table)
    
    story.append(PageBreak())
    
    # SECTION 8: VALIDATION EVIDENCE
    story.append(Paragraph("8. Validation Evidence", h1_style))
    
    story.append(Paragraph("""
    This strategy passed 5 out of 6 rigorous falsification tests:
    """, body_style))
    
    val_data = [
        ['Test', 'Result', 'Detail'],
        ['Active Return p-value', '✓ PASS', 'p=0.0002'],
        ['Pre/Post Consistency', '✓ PASS', 'IR: 1.81 / 0.86'],
        ['Proxy Falsification', '✓ PASS', 'Real > Random'],
        ['Bootstrap CI', '✗ FAIL', 'CIs overlap'],
        ['Monte Carlo', '✓ PASS', 'p=0.0000'],
        ['Walk-Forward', '✓ PASS', '3/3 folds'],
    ]
    val_table = Table(val_data, colWidths=[2*inch, 1.2*inch, 2*inch])
    val_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5f7a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TEXTCOLOR', (1, 1), (1, 3), colors.HexColor('#28a745')),
        ('TEXTCOLOR', (1, 4), (1, 4), colors.HexColor('#dc3545')),
        ('TEXTCOLOR', (1, 5), (1, 6), colors.HexColor('#28a745')),
    ]))
    story.append(val_table)
    
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("""
    <b>Note:</b> This is the only strategy out of 35+ tested that achieved this level of validation.
    The strategy is validated but not guaranteed - past performance does not guarantee future results.
    """, important_style))
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    try:
        path = create_setup_guide()
        print(f"\n📄 Open the PDF at:\n   {path}")
    except ImportError:
        print("❌ Need reportlab: pip install reportlab")
