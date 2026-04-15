import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

class SharedLedger:
    """
    Mock implementation of a Shared Ledger (Blockchain) for chemical purity verification.
    In a real-world scenario, this would connect to a DLT node (e.g., Ethereum, Hyperledger).
    """

    def __init__(self):
        # Mock Data Store: {VendorID: {ItemNumber: Purity%}}
        # These are "Trusted" vendors who publish purity certificates on-chain.
        self._ledger_db = {
            "VEND001": {"CHEM-A-100": 99.5, "CHEM-B-50": 50.0}, # Trusted Vendor 1
            "GLOBAL_CHEM": {"SOL-99": 99.9},                    # Global Chemical Corp
            "PURE_SUPPLY": {"ACID-SUL-98": 98.0, "UREA-46": 46.0},
        }
        LOGGER.info("Shared Ledger initialized with mock data.")

    def get_verified_purity(self, vendor_id: str, item_number: str) -> float | None:
        """
        Queries the transparent ledger for a verified purity claim.
        
        Args:
            vendor_id: The ID of the vendor selling the item.
            item_number: The item number/SKU.
            
        Returns:
            float: Verified purity percentage (0-100) if found.
            None: If no verified record exists on the ledger.
        """
        vendor_id = vendor_id.strip() if vendor_id else ""
        item_number = item_number.strip() if item_number else ""
        
        # Check if Vendor is on the ledger
        if vendor_id in self._ledger_db:
            vendor_ledger = self._ledger_db[vendor_id]
            # Check if likely item match (exact for now)
            if item_number in vendor_ledger:
                purity = vendor_ledger[item_number]
                LOGGER.info(f"Ledger VERIFIED: {vendor_id} | {item_number} = {purity}%")
                return purity
        
        LOGGER.debug(f"Ledger Lookup Miss: {vendor_id} | {item_number}")
        return None

    def add_verified_record(self, vendor_id: str, item_number: str, purity: float):
        """Adds a record to the mock ledger (for testing)."""
        if vendor_id not in self._ledger_db:
            self._ledger_db[vendor_id] = {}
        self._ledger_db[vendor_id][item_number] = purity
        LOGGER.info(f"Record added to Ledger: {vendor_id} | {item_number} = {purity}%")
