"""
Category Mapper - Simple and clean category normalization.

This module normalizes category names across datasets using minimal rules:
1. Normalize case (UPPERCASE → Title Case)
2. Replace underscores with spaces
3. Map only truly identical categories
4. Keep general categories (ENGINEERING, TEACHER, etc.) as-is
"""

from typing import Dict, Optional, List


# Only map categories that are TRULY the same thing with different formatting
SIMPLE_MAPPINGS: Dict[str, str] = {
    # Case variations of the same role
    "hr": "HR",
    "Hr": "HR",
    "pmo": "PMO",
    "Pmo": "PMO",
    "sap developer": "SAP Developer",

    # Formatting variations (underscores/spaces)
    "Software_Developer": "Software Developer",
    "Python_Developer": "Python Developer",
    "Java_Developer": "Java Developer",
    "Web_Developer": "Web Developer",
    "Front_End_Developer": "Front End Developer",
    "Database_Administrator": "Database Administrator",
    "Systems_Administrator": "Systems Administrator",
    "Network_Administrator": "Network Administrator",
    "Security_Analyst": "Security Analyst",
    "Project_Manager": "Project Manager",

    # Minor name variations that mean exactly the same thing
    "Health and fitness": "Fitness",
    "Web Designing": "Designer",
    "Electrical Engineering": "Engineering",  # Merge specific engineering into general
    "Civil Engineer": "Engineering",
    "Mechanical Engineer": "Engineering",

    # .NET variations
    ".NET Developer": "DotNet Developer",
    ".Net Developer": "DotNet Developer",
    "Dotnet Developer": "DotNet Developer",

    # Devops case
    "Devops Engineer": "DevOps Engineer",
    "devops engineer": "DevOps Engineer",

    # Data science case
    "data science": "Data Science",
    "Data science": "Data Science",
}

#TODO: this is the same normalization logic as the processing
def normalize_category(category: str) -> str:
    """
    Normalize a category string with simple rules.

    Steps:
    1. Strip whitespace
    2. Check direct mapping (as-is, case-sensitive)
    3. Replace underscores with spaces
    4. Check mapping again
    5. If all uppercase → Title Case
    6. Return result

    Args:
        category: Raw category string

    Returns:
        Normalized category string
    """
    if not category or not isinstance(category, str):
        return "Unknown"

    # Step 1: Strip whitespace
    category = category.strip()

    if not category:
        return "Unknown"

    # Step 2: Direct mapping check (exact match)
    if category in SIMPLE_MAPPINGS:
        return SIMPLE_MAPPINGS[category]

    # Step 3: Replace underscores with spaces
    normalized = category.replace("_", " ")

    # Step 4: Check mapping again after underscore replacement
    if normalized in SIMPLE_MAPPINGS:
        return SIMPLE_MAPPINGS[normalized]

    # Step 5: If all uppercase, convert to title case
    if normalized.isupper():
        normalized = normalized.title()

    # Step 6: Final mapping check (after title case)
    if normalized in SIMPLE_MAPPINGS:
        return SIMPLE_MAPPINGS[normalized]

    return normalized

#TODO: WHY THIS EXIST REMOVE IT
def map_category(category: str, resume_text: Optional[str] = None) -> str:
    """
    Map a category to its normalized form.

    This is the main entry point for category normalization.

    Args:
        category: Raw category string
        resume_text: Optional resume text (IGNORED - no keyword-based classification)

    Returns:
        Normalized category name
    """
    # Just normalize - no keyword-based classification nonsense
    return normalize_category(category)


def get_all_unique_categories(categories: List[str]) -> List[str]:
    """
    Get all unique normalized categories from a list.

    Args:
        categories: List of raw category strings

    Returns:
        Sorted list of unique normalized categories
    """
    normalized = [normalize_category(cat) for cat in categories]
    return sorted(set(normalized))


def print_category_mapping_report(categories: List[str]) -> None:
    """
    Print a report showing which categories were mapped.

    Args:
        categories: List of raw category strings
    """
    from collections import Counter

    print("\n" + "="*70)
    print("CATEGORY NORMALIZATION REPORT")
    print("="*70)

    # Track transformations
    transformations: Dict[str, str] = {}

    for cat in set(categories):
        normalized = normalize_category(cat)
        if cat != normalized:
            transformations[cat] = normalized

    # Count before/after
    original_count = len(set(categories))
    normalized_cats = [normalize_category(c) for c in categories]
    normalized_count = len(set(normalized_cats))

    print(f"\nCategories before normalization: {original_count}")
    print(f"Categories after normalization:  {normalized_count}")
    print(f"Reduction: {original_count - normalized_count}")

    # Print transformations
    if transformations:
        print(f"\nCategory Transformations ({len(transformations)}):")
        for orig, norm in sorted(transformations.items()):
            orig_count = categories.count(orig)
            print(f"  '{orig}' → '{norm}' ({orig_count} samples)")
    else:
        print("\nNo transformations applied - all categories already normalized")

    # Print final distribution
    distribution = Counter(normalized_cats)
    print(f"\nFinal Category Distribution ({len(distribution)} categories):")
    for cat, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * min(count // 50, 30)
        print(f"  {cat:35s} {count:5d} {bar}")

    print("="*70)
