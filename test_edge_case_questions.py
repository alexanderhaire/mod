"""
Proactive edge case testing: Questions that might stump the model.
This file contains a comprehensive test suite of potentially problematic questions.
"""

# Categories of edge cases to test
EDGE_CASE_QUESTIONS = {
    "ambiguous_time_references": [
        "What were sales last month?",  # needs to calculate which month is "last month"
        "Show me inventory for Q4",  # needs to understand Q4 = Oct,Nov,Dec
        "YTD sales for SOARBLM02",  # Year-to-date - needs current date
        "Sales for the past 3 months",  # rolling 3-month window
        "Compare this quarter to last quarter",  # needs quarter logic
    ],
    
    "negation_questions": [
        "What items are NOT in the BOM for SOARBLM02?",  # SQL NOT IN
        "Show customers without orders",  # LEFT JOIN with NULL check
        "Items that have never been ordered",  # complex negation
        "Items that don't have inventory",  # negation with zero/null
        "Exclude ITEM123 from results",  # explicit exclusion
    ],
    
    "null_handling": [
        "Items with missing cost information",  # WHERE cost IS NULL
        "Show items where standard cost is null",  # explicit NULL check
        "What items don't have a description?",  # NULL descriptions
        "Items with blank lot numbers",  # NULL vs empty string
    ],
     "case_sensitivity": [
        "What is our usage on npkacek?",  # lowercase item code
        "Sales for soarblm02",  # lowercase - needs UPPER() or collation
        "Items matching 'item'",  # case-insensitive search
    ],
    
    "cost_term_confusion": [
        "What is the current cost of SOARBLM02?",  # current vs standard vs last
        "Show standard cost vs current cost",  # comparison of cost fields
        "Items where last cost differs from standard cost",  # field comparison
    ],
    
    "percentage_calculations": [
        "What percentage of total sales is SOARBLM02?",  # ratio calculation
        "Show top 10% of items by sales",  # percentile
        "Items with more than 20% margin",  # percentage comparison
    ],
    
    "year_over_year": [
        "Compare sales this year vs last year",  # YoY comparison
        "What's the growth rate from 2024 to 2025?",  # growth calculation
        "Show same month last year",  # temporal offset
    ],
    
    "unit_conversions": [
        "Convert gallons to liters",  # unit conversion
        "Show weight in pounds and kilograms",  # dual units
    ],
    
    "ambiguous_aggregations": [
        "Average of monthly averages",  # double aggregation edge case
        "Total of totals",  # could be misleading
        "Sum of percentages",  # may not make sense
    ],
    
    "multi_step_logic": [
        "Find items with declining sales AND low inventory",  # multiple conditions
        "Items with high usage but no open POs",  # join + filter
        "Top customers for items in short supply",  # nested query
    ],
    
    "typos_and_misspellings": [
        "What is uasge on NPKACEK?",  # "uasge" instead of "usage"
        "Sales for SOARDLM02",  # wrong item code
        "Show purhcase orders",  # "purhcase" instead of "purchase"
    ],
    
    "vague_questions": [
        "Show me some data",  # extremely vague
        "What's going on with inventory?",  # no specific metric
        "Tell me about sales",  # no timeframe or item
    ],
    
    "comparison_operators": [
        "Items where qty on hand less than order point",  # < operator
        "Items where sales greater than 100",  # > operator
        "Items equal to ITEM123",  # = operator (should use string match)
    ],
    
    "date_edge_cases": [
        "Sales on February 29, 2024",  # leap year
        "December 31 sales",  # year-end
        "First day of the month",  # ambiguous which month
    ],
}

# Expected behaviors/fixes needed
EXPECTED_FIXES = {
    "ambiguous_time_references": "Need to resolve relative dates to absolute dates using current date",
    "negation_questions": "Need to generate proper NOT IN / NOT EXISTS / LEFT JOIN NULL patterns",
    "null_handling": "Need to distinguish NULL vs zero vs empty string",
    "case_sensitivity": "Need to use UPPER() or case-insensitive collation",
    "cost_term_confusion": "Need to clarify which cost field (STNDCOST, CURRCOST, LSTCOST)",
    "percentage_calculations": "Need to generate proper CAST/DECIMAL for division",
    "year_over_year": "Need to generate self-join or window functions for temporal comparison",
    "unit_conversions": "Need to either reject or provide conversion factors",
    "ambiguous_aggregations": "Need to validate if aggregation makes sense",
    "multi_step_logic": "May need reasoning coordinator for complex queries",
    "typos_and_misspellings": "Need fuzzy matching or spell correction",
    "vague_questions": "Need to ask clarifying questions or provide common metrics",
    "comparison_operators": "Need to map natural language to SQL operators correctly",
    "date_edge_cases": "Need proper date handling including leap years",
}

if __name__ == "__main__":
    print("Edge Case Test Questions")
    print("=" * 80)
    for category, questions in EDGE_CASE_QUESTIONS.items():
        print(f"\n{category.upper().replace('_', ' ')}")
        print("-" * 80)
        for q in questions:
            print(f"  • {q}")
        print(f"\nExpected fix: {EXPECTED_FIXES.get(category, 'TBD')}")
