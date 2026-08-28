import re
from typing import Optional, Any

# Number word to integer mapping
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}

def parse_number(text: str) -> Optional[float]:
    """Parse integer, float, or word number from text."""
    text_clean = text.strip().lower()
    if text_clean in NUMBER_WORDS:
        return float(NUMBER_WORDS[text_clean])
    try:
        return float(text_clean)
    except ValueError:
        return None

def extract_quantitative_evidence(text: str, target_keyword: Optional[str] = None) -> Optional[dict[str, Any]]:
    """
    Extract structured quantitative evidence from conversational text.
    Returns:
        dict with value, unit, period, actor, evidence or None if no numeric evidence exists.
    """
    # Check actor / subject
    is_third_party = bool(re.search(r"\b(my\s+(brother|sister|friend|boss|coworker|mother|father)|he|she|they)\b", text, re.IGNORECASE))
    is_first_person = bool(re.search(r"\b(i|my|we|our)\b", text, re.IGNORECASE))
    actor = "third_party" if (is_third_party and not re.search(r"\b(i|me|my)\s+(work|practice|drink|commute)", text, re.IGNORECASE)) else "speaker"

    # Pattern for numeric expressions like:
    # "five hours every week", "5 hours/week", "3 cups of coffee every morning", "two hours every day"
    pattern = re.compile(
        r"(?P<val>\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty)\s+"
        r"(?P<unit>hours?|hrs?|minutes?|mins?|cups?|mugs?|km|kilometers?|miles?|times?|repetitions?|reps?|sessions?|days?|months?|years?)\s*"
        r"(?:of\s+[a-zA-Z\s]+?)?\s*"
        r"(?:(?:every|each|per|a|\/)\s*(?P<period>week|day|morning|night|evening|month|year|shift))?",
        re.IGNORECASE
    )

    for match in pattern.finditer(text):
        raw_val = match.group("val")
        unit = match.group("unit").lower()
        period = (match.group("period") or "").lower()
        if not period:
            if "morning" in text.lower():
                period = "day"
            elif "week" in text.lower():
                period = "week"
            elif "day" in text.lower():
                period = "day"
        
        numeric_val = parse_number(raw_val)
        if numeric_val is not None:
            return {
                "value": numeric_val,
                "unit": unit,
                "period": period,
                "actor": actor,
                "evidence": match.group(0).strip(),
                "full_text": text.strip(),
            }

    return None

test_snippets = [
    "I practice yoga for five hours every week.",
    "I drink three cups of coffee every morning.",
    "I commute for two hours every day.",
    "I work eight hours every day.",
    "My brother works twelve hours every day.",
    "I spend a lot of time outside.",
    "Traffic has been getting worse every month.",
]

for s in test_snippets:
    extracted = extract_quantitative_evidence(s)
    print(f"Snippet: '{s}'")
    print(f"  Extracted: {extracted}\n")
