"""
Reasoning Coordinator - Multi-Step Question Decomposition and Execution

This module enables the system to handle complex multi-step questions by:
1. Decomposing the question into logical sub-steps
2. Executing each step sequentially
3. Accumulating context across steps
4. Synthesizing final results
"""

import datetime
import json
import logging
from typing import Any

import pyodbc
import requests

from constants import LOGGER, OPENAI_CHAT_URL, OPENAI_DEFAULT_MODEL
from handlers import handle_custom_sql_question
from secrets_loader import load_openai_settings
from sql_utils import _extract_json_block


def call_openai_decompose_question(
    prompt: str,
    today: datetime.date,
    context: dict | None = None
) -> dict | None:
    """
    Use LLM to decompose a complex question into sequential sub-steps.
    
    Returns:
        dict with:
            - steps: list of step objects with 'description', 'type', 'dependencies'
            - reasoning: explanation of decomposition
            - is_multi_step: boolean
    """
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key:
        return None
    
    model = settings.get("model", OPENAI_DEFAULT_MODEL)
    
    system_prompt = (
        "You are a query planner for an ERP data analysis system. "
        "Your job is to determine if a question requires multiple sequential steps to answer, and if so, decompose it. "
        "A multi-step question requires executing one sub-query, using its results to inform the next query. "
        "\n\nExamples of multi-step questions:\n"
        "- 'Find items with declining sales, then show their BOM components' (2 steps: identify items, then get BOMs)\n"
        "- 'What finished goods use NPK3011, and what's their demand forecast?' (2 steps: BOM where-used, then demand)\n"
        "- 'Calculate which raw materials are running low, then show what finished goods will be impacted' (2 steps: low stock items, then reverse BOM)\n"
        "\n\nExamples of single-step questions (even if complex):\n"
        "- 'If SOARBLM02 demand increases 3%, what raw materials to buy?' (handled by what-if handler)\n"
        "- 'Compare usage of NPK3011 vs NPKACEK' (single complex SQL)\n"
        "- 'Show top selling items with their BOM components' (single SQL with joins)\n"
        "\n\nReturn JSON with:\n"
        "{\n"
        "  'is_multi_step': true/false,\n"
        "  'reasoning': 'explanation of why',\n"
        "  'steps': [\n"
        "    {'step_num': 1, 'description': 'what to do', 'type': 'sql_query'|'analysis'|'calculation', 'output': 'what this produces'},\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "If is_multi_step is false, steps can be empty or have just 1 step."
    )
    
    user_sections = [
        f"Current date: {today.isoformat()}",
        f"Question: {prompt.strip()}",
    ]
    if context:
        user_sections.append(f"Context: {json.dumps(context)}")
    
    user_sections.append("Determine if this requires multiple sequential steps. Respond with JSON only.")
    
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_sections)}
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning(f"Question decomposition failed: {err}")
        return None


def execute_reasoning_chain(
    cursor: pyodbc.Cursor,
    prompt: str,
    today: datetime.date,
    context: dict | None = None,
    max_iterations: int = 5
) -> dict:
    """
    Execute a multi-step reasoning chain by:
    1. Decomposing the question into steps
    2. Executing each step sequentially
    3. Accumulating context from each step
    4. Synthesizing final results
    
    Returns:
        dict with 'data', 'insights', 'reasoning_steps', etc.
    """
    context = context or {}
    
    # Step 1: Decompose the question
    decomposition = call_openai_decompose_question(prompt, today, context)
    
    if not decomposition or not decomposition.get("is_multi_step"):
        # Not a multi-step question, return None to let regular handlers take over
        LOGGER.info("Question does not require multi-step reasoning")
        return None
    
    steps = decomposition.get("steps", [])
    if not steps or len(steps) < 2:
        LOGGER.info("Decomposition returned <2 steps, not treating as multi-step")
        return None
    
    LOGGER.info(f"Decomposed into {len(steps)} steps: {decomposition.get('reasoning')}")
    
    # Step 2: Execute each step sequentially
    accumulated_context = context.copy()
    step_results = []
    
    for i, step in enumerate(steps[:max_iterations], 1):
        step_description = step.get("description", "")
        step_type = step.get("type", "sql_query")
        
        LOGGER.info(f"Executing step {i}/{len(steps)}: {step_description}")
        
        # Execute the step as a sub-query
        try:
            # Create a modified prompt for this specific step
            step_prompt = step_description
            
            # Add context from previous steps
            if step_results:
                # Summarize previous results for context
                prev_summary = f"Previous step results: "
                for prev_result in step_results:
                    if prev_result.get("data"):
                        row_count = len(prev_result["data"])
                        prev_summary += f"{prev_result.get('step_description', 'Step')} returned {row_count} rows. "
                accumulated_context["previous_steps"] = prev_summary
            
            # Execute using existing handler
            step_result = handle_custom_sql_question(
                cursor, 
                step_prompt, 
                today, 
                accumulated_context
            )
            
            if step_result:
                step_result["step_num"] = i
                step_result["step_description"] = step_description
                step_results.append(step_result)
                
                # Accumulate key information
                if step_result.get("data"):
                    # Store first few rows for context in next step
                    accumulated_context[f"step_{i}_sample"] = step_result["data"][:5]
                if step_result.get("entities"):
                    accumulated_context.update(step_result["entities"])
            else:
                # Step failed, log and continue
                LOGGER.warning(f"Step {i} returned no result")
                step_results.append({
                    "step_num": i,
                    "step_description": step_description,
                    "error": "No result returned"
                })
        
        except Exception as err:
            LOGGER.error(f"Step {i} failed with error: {err}")
            step_results.append({
                "step_num": i,
                "step_description": step_description,
                "error": str(err)
            })
    
    # Step 3: Synthesize final results
    if not step_results:
        return {"insights": {"summary": "Multi-step reasoning attempted but no steps completed successfully."}}
    
    # Use the last successful step's data as the primary result
    final_result = None
    for result in reversed(step_results):
        if result.get("data"):
            final_result = result
            break
    
    if not final_result:
        final_result = step_results[-1]
    
    # Enhance with reasoning chain information
    reasoning_summary = f"Multi-step analysis completed in {len(step_results)} steps:\n"
    for result in step_results:
        step_num = result.get("step_num")
        step_desc = result.get("step_description")
        if result.get("error"):
            reasoning_summary += f"  {step_num}. {step_desc} - ERROR: {result['error']}\n"
        else:
            row_count = len(result.get("data", []))
            reasoning_summary += f"  {step_num}. {step_desc} - {row_count} results\n"
    
    final_result["insights"] = final_result.get("insights", {})
    final_result["insights"]["reasoning_chain"] = reasoning_summary
    final_result["insights"]["multi_step"] = True
    final_result["reasoning_steps"] = step_results
    
    return final_result


def should_use_reasoning_coordinator(prompt: str) -> bool:
    """
    Quick heuristic to determine if a question might benefit from reasoning coordination.
    Returns True if the prompt contains indicators of multi-step logic.
    """
    lower = prompt.lower()
    
    # Multi-step indicators
    multi_step_keywords = (
        "then show",
        "then calculate",
        "then find",
        "and then",
        "first find",
        "next",
        "after that",
        "followed by",
        "impact on",
        "affected by",
        "which items use",
        "what uses",
        "reverse bom",
        "where used"
    )
    
    return any(keyword in lower for keyword in multi_step_keywords)
