from collections.abc import Mapping


def build_bom_guidance(prompt: str, plan: Mapping | None, data_rows: list[dict]) -> str | None:
    """Provide a Bill of Materials note when the user is asking about components or BOM tables are used."""
    prompt_text = prompt.lower() if isinstance(prompt, str) else ""
    bom_keywords = ("bill of materials", "bom", "made of", "components", "what is", "made from", "composition")
    asked_for_bom = any(keyword in prompt_text for keyword in bom_keywords)

    plan = plan or {}
    sql_text = plan.get("sql", "") if isinstance(plan, Mapping) else ""
    references_bom_table = any(token in sql_text.upper() for token in ("BM00111", "BM010115", "BM00101"))

    if not (asked_for_bom or references_bom_table):
        return None

    entities = plan.get("entities") if isinstance(plan, Mapping) else {}
    item = entities.get("item") if isinstance(entities, Mapping) else None
    if not item and data_rows:
        sample = data_rows[0]
        if isinstance(sample, Mapping):
            item = sample.get("ITEMNMBR") or sample.get("ParentItem") or sample.get("Parent")

    guidance = "Bill of Materials page: Inventory > Inquiry > Bill of Materials shows the component structure"
    if item:
        guidance += f" for {item}"
    guidance += "."
    if not data_rows:
        guidance += " No component rows were returned; check the BOM screen to confirm whether the item has a defined bill or is stored under another site."
    return guidance
