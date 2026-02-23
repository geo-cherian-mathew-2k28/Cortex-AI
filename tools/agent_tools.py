"""
Lexi-Sense Agent Tools
Specialized tools the AI agent can invoke for calculations, table generation,
document comparison, and CSV export.
"""
import re
import json
import csv
import io
from typing import List, Dict, Any, Optional


# ═══════════════════════════════════════════════════════════════════════
#  CALCULATOR TOOL
# ═══════════════════════════════════════════════════════════════════════
def calculator_tool(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports basic arithmetic, percentages, and common math functions.
    """
    # Sanitize input
    allowed_chars = set("0123456789+-*/().%, ")
    clean = expression.replace("^", "**").replace("×", "*").replace("÷", "/")

    # Handle percentages like "20% of 500"
    pct_match = re.match(r'([\d.]+)%\s*of\s*([\d.]+)', clean)
    if pct_match:
        pct = float(pct_match.group(1))
        base = float(pct_match.group(2))
        result = (pct / 100) * base
        return f"{pct}% of {base} = {result:,.2f}"

    # Handle percentage change "from X to Y"
    change_match = re.match(r'.*?from\s*([\d,.]+)\s*to\s*([\d,.]+)', expression, re.IGNORECASE)
    if change_match:
        old = float(change_match.group(1).replace(",", ""))
        new = float(change_match.group(2).replace(",", ""))
        change = ((new - old) / old) * 100
        return f"Change from {old:,.2f} to {new:,.2f} = {change:+.2f}%"

    # Clean for eval
    clean = re.sub(r'[^\d+\-*/().%\s]', '', clean)
    clean = clean.replace("%", "/100")

    try:
        result = eval(clean, {"__builtins__": {}}, {})
        return f"{expression} = {result:,.4f}" if isinstance(result, float) else f"{expression} = {result:,}"
    except Exception as e:
        return f"Could not compute: {expression}. Error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════
#  TABLE GENERATOR TOOL
# ═══════════════════════════════════════════════════════════════════════
def table_generator_tool(data: List[Dict[str, Any]], title: str = "") -> str:
    """
    Generate a markdown-formatted table from structured data.
    Input: List of dictionaries with consistent keys.
    """
    if not data:
        return "No data available to create a table."

    headers = list(data[0].keys())

    # Calculate column widths
    widths = {h: len(str(h)) for h in headers}
    for row in data:
        for h in headers:
            val = str(row.get(h, ""))
            widths[h] = max(widths[h], len(val))

    # Build table
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")

    # Header
    header_line = "| " + " | ".join(str(h).ljust(widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * widths[h] for h in headers) + " |"
    lines.append(header_line)
    lines.append(sep_line)

    # Rows
    for row in data:
        row_line = "| " + " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers) + " |"
        lines.append(row_line)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  COMPARISON TOOL
# ═══════════════════════════════════════════════════════════════════════
def comparison_tool(doc_a_summary: str, doc_b_summary: str, file_a: str, file_b: str) -> str:
    """
    Generate a structured comparison prompt between two document excerpts.
    Returns a formatted comparison template that the LLM can fill in.
    """
    template = f"""## 📊 Document Comparison: {file_a} vs {file_b}

### Document A: {file_a}
{doc_a_summary[:2000]}

### Document B: {file_b}
{doc_b_summary[:2000]}

---
Please analyze these documents and provide:
1. **Key Similarities** – What both documents share
2. **Key Differences** – Where they diverge
3. **Conflicting Information** – Any contradictions found
4. **Unique Content** – Information present in only one document
5. **Summary Assessment** – Overall comparison verdict
"""
    return template


# ═══════════════════════════════════════════════════════════════════════
#  CSV EXPORT TOOL
# ═══════════════════════════════════════════════════════════════════════
def csv_export_tool(data: List[Dict[str, Any]]) -> str:
    """
    Convert structured data to CSV format string.
    Returns the CSV content that can be downloaded by the frontend.
    """
    if not data:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()


# ═══════════════════════════════════════════════════════════════════════
#  ENTITY EXTRACTOR (helper for summaries)
# ═══════════════════════════════════════════════════════════════════════
def extract_entities_prompt(text: str) -> str:
    """Generate a prompt to extract named entities from text."""
    return f"""Analyze the following text and extract:
1. **Dates** – All dates and timeframes mentioned
2. **Names** – People, organizations, companies
3. **Monetary Values** – Financial figures, prices, amounts
4. **Locations** – Places, addresses
5. **Key Terms** – Domain-specific important terms

Text (first 3000 chars):
{text[:3000]}

Provide the extracted entities in a structured format."""


# ═══════════════════════════════════════════════════════════════════════
#  TOOL REGISTRY
# ═══════════════════════════════════════════════════════════════════════
TOOL_DEFINITIONS = {
    "calculator": {
        "name": "calculator",
        "description": "Perform mathematical calculations, percentages, and financial computations",
        "function": calculator_tool,
    },
    "table_generator": {
        "name": "table_generator",
        "description": "Generate formatted tables from structured data",
        "function": table_generator_tool,
    },
    "comparison": {
        "name": "comparison",
        "description": "Compare two documents and highlight similarities and differences",
        "function": comparison_tool,
    },
    "csv_export": {
        "name": "csv_export",
        "description": "Export structured data as downloadable CSV",
        "function": csv_export_tool,
    },
}
