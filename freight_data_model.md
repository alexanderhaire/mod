# Structured Freight Data Model

Based on ERP analysis, here's the structured data for freight prediction.

## Route Definitions

| Route ID | Origin | Destination | Material Vendor | Freight Carrier | Rate Type | Benchmark Rate |
|----------|--------|-------------|-----------------|-----------------|-----------|----------------|
| `MUL-PC` | Mulberry, FL (33860) | Plant City, FL | Various | David Cole | Per Ton | $11.50/ton |
| `CAPE-PC` | Cape Canaveral, FL (32920) | Plant City, FL | SQM (via port) | David Cole / Jesse Cole | Per Ton | $28.00/ton |
| `ATL-PC` | Atlanta, GA (30374) | Plant City, FL | SQM North American | David Cole | Per Ton | ~$75/ton |
| `MI-PC` | West Bloomfield, MI (48325) | Plant City, FL | The Bottle Crew | PLS Logistics | Flat Rate | ~$3,000/load |

## Key Vendors

### Material Suppliers
| Vendor ID | Name | Location | Primary Items |
|-----------|------|----------|---------------|
| `SQM` | SQM North American Corp. | Atlanta, GA 30374 | NPKKNO3 |
| `BOTTLECREW` | The Bottle Crew | West Bloomfield, MI 48325 | ZZ2.5GALFFL, ZZ2.5GALTALL |

### Freight Carriers
| Vendor ID | Name | Location | Typical Routes |
|-----------|------|----------|----------------|
| `DAVIDCOLE` | David Cole Trucking, Inc. | Lithia, FL | Cape, Atlanta, Mulberry |
| `JESSE COLE TRUC` | Jesse Cole Trucking Inc | Lithia, FL | Cape, Regional |
| `PLS` | PLS Logistics Services | Chicago, IL | Long haul (Michigan, Ohio) |

## Data Relationships

```
Material PO (Bottle Crew)     Freight Invoice (PLS)
---------------------------   -------------------------
RECEIPTDATE: 2025-12-17  -->  DOCDATE: ~2025-12-17
ITEMNMBR: ZZ2.5GALFFL         DOCAMNT: ~$3,000
SUBTOTAL: $11,894.40          (matched by date proximity)
ORFRTAMT: $0 (separate)
```

## Rate Estimation Logic

```python
def estimate_freight_cost(origin_zip, tonnage=None, load_type="standard"):
    # Flat rate routes (packaging, long haul)
    if origin_zip.startswith("48"):  # Michigan
        return 3000.00  # Flat rate
    
    # Per-ton routes
    rates = {
        "33860": 11.50,   # Mulberry
        "32920": 28.00,   # Cape Canaveral
        "30374": 75.00,   # Atlanta (est)
    }
    
    rate = rates.get(origin_zip, 20.00)  # Default $20/ton
    
    if tonnage:
        return rate * tonnage
    return rate  # Return per-ton rate if no tonnage

# Examples:
# estimate_freight_cost("32920", tonnage=29) = $812.00
# estimate_freight_cost("48325") = $3000.00 (flat)
```

## Suggested Enhancements

1. **Link Material POs to Freight Invoices** - Match by date range (+/- 3 days)
2. **Track Carrier Performance** - On-time %, damage rate
3. **Seasonal Patterns** - Fuel surcharges, peak season rates
