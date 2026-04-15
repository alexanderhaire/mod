
from fpdf import FPDF
import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Golden Omni Strategy Report (Florida Edition v2)', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

def create_pdf():
    pdf = PDF()
    pdf.add_page()
    
    # Title Info
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'OFFICIAL BROKERAGE GUIDE (FLORIDA)', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Generated: {datetime.date.today()}', 0, 1, 'C')
    pdf.ln(10)
    
    # Strategy Recap
    pdf.chapter_title('1. Strategy Recap (Weekly Rebalance)')
    pdf.chapter_body(
        "The Golden Omni uses a weekly rotation between Growth (SPY/Crypto), Deflation Safety (TLT), "
        "and Inflation Safety (Energy/Gold). Because you are in Florida, we recommend the 'ETF Proxy' "
        "method to ensure full compliance and easy execution.\n\n"
        "ETF Proxy Map:\n"
        "- Bitcoin -> IBIT (iShares Bitcoin Trust)\n"
        "- Ethereum -> ETHE (Grayscale Ethereum Trust)\n"
    )
    
    # Broker Comparison
    pdf.chapter_title('2. Brokerage Showdown: The Florida Options')
    pdf.chapter_body(
        "Based on your request, here is the deep dive into Schwab and Webull vs the competitions.\n\n"
    )
    
    # Charles Schwab
    pdf.chapter_title('3. Charles Schwab (The "Boomer" Giant)')
    pdf.chapter_body(
        "VERDICT: Excellent for Manual, Difficult for Automation.\n\n"
        "PROS:\n"
        "- **Stability:** Trillions in assets, very safe.\n"
        "- **ETFs:** ZERO commissions on all required ETFs (SPY, TLT, XLE, GLD, IBIT, ETHE).\n"
        "- **Research:** Best-in-class tools and charts.\n\n"
        "CONS:\n"
        "- **API:** They have a 'Trader API' but it is complex, aimed at institutions, and requires a lengthy approval process compared to Alpaca/IBKR.\n"
        "- **Crypto:** NO direct crypto. You MUST use the IBIT/ETHE proxies (which is fine for this strategy).\n\n"
        "RECOMMENDATION: Use Schwab if you plan to trade MANUALLY every Friday."
    )
    
    # Webull
    pdf.chapter_title('4. Webull (The "Retail" Challenger)')
    pdf.chapter_body(
        "VERDICT: Good Hybrid, but risky automation.\n\n"
        "PROS:\n"
        "- **User Exp:** Great mobile app, very modern.\n"
        "- **Crypto:** Supports direct crypto (BTC/ETH/SOL/DOGE) via Webull Pay.\n"
        "- **Florida:** Fully available.\n\n"
        "CONS:\n"
        "- **API:** exist but documentation is community-maintained (unofficial python wrappers) and less reliable than Alpaca.\n"
        "- **Spreads:** Crypto spreads on Webull can be wider than Coinbase/Binance.\n\n"
        "RECOMMENDATION: Use Webull if you want a great mobile app for Manual trading and want meaningful access to Altcoins (SOL/DOGE) that Schwab doesn't have."
    )
    
    # The Winner
    pdf.chapter_title('5. The Automation Winner: Alpaca')
    pdf.chapter_body(
        "If you want to 'Set It and Forget It' with a Python script, **Alpaca** remains the only viable choice for a retail trader in Florida.\n\n"
        "- **Schwab's** API is too hard to get.\n"
        "- **Webull's** API is unofficial/finicky.\n"
        "- **Robinhood** has no official trading API.\n\n"
        "**Alpaca + IBIT/ETHE** is the path of least resistance."
    )

    filename = 'Golden_Omni_Broker_Guide.pdf'
    pdf.output(filename, 'F')
    print(f"PDF generated: {filename}")

if __name__ == "__main__":
    create_pdf()
