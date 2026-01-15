"""
Category normalization utilities for resume classification.

This module provides unified category normalization functionality,
consolidating logic from category_mapper.py and dataset_cleaner.py.
"""

from typing import Dict
from src.category_hierarchy import is_valid_category


# Consolidated mapping dictionary combining both sources
CATEGORY_MAPPINGS: Dict[str, str] = {
    # From SIMPLE_MAPPINGS (category_mapper.py)
    # Software Engineering variations
    "data engineer": "Data Engineer", 
    "Data engineer": "Data Engineer",
    "devops engineer": "DevOps Engineer",
    "Devops engineer": "DevOps Engineer", 
    "DevOps engineer": "DevOps Engineer",
    "ml engineer": "ML Engineer",
    "ML engineer": "ML Engineer",
    "software engineer": "Software Engineer",
    "Software engineer": "Software Engineer",
    "frontend developer": "Front End Developer",
    "Frontend developer": "Front End Developer",
    "backend developer": "Backend Developer",
    "Backend developer": "Backend Developer",
    "full stack developer": "Full Stack Developer",
    "Full stack developer": "Full Stack Developer",
    "fullstack developer": "Full Stack Developer",
    "Fullstack developer": "Full Stack Developer",
    "mobile developer": "Mobile Developer",
    "Mobile developer": "Mobile Developer",
    # Data science case
    "data science": "Data Science",
    "Data science": "Data Science",
    
    # From CATEGORY_NORMALIZATION (dataset_cleaner.py)
    # Resume.csv UPPERCASE → hierarchy names
    "ACCOUNTANT": "Accountant",
    "ADVOCATE": "Advocate", 
    "AGRICULTURE": "Agriculture",
    "APPAREL": "Apparel",
    "ARTS": "Arts",
    "AUTOMOBILE": "Automobile",
    "AVIATION": "Aviation",
    "BANKING": "Banking",
    "BPO": "Bpo",
    "BUSINESS-DEVELOPMENT": "Business-Development",
    "CHEF": "Chef",
    "CONSTRUCTION": "Construction",
    "CONSULTANT": "Consultant",
    "DESIGNER": "Designer",
    "DIGITAL-MEDIA": "Digital-Media",
    "ENGINEERING": "Engineering",
    "FINANCE": "Finance",
    "FITNESS": "Fitness",
    "HEALTHCARE": "Healthcare",
    "HR": "HR",
    "INFORMATION-TECHNOLOGY": "Information-Technology",
    "PUBLIC-RELATIONS": "Public-Relations",
    "SALES": "Sales",
    "TEACHER": "Teacher",
    # UpdatedResumeDataSet.csv variations
    "Health and fitness": "Fitness",
    "Web Designing": "Designer",
    "Electrical Engineering": "Engineering",
    "Civil Engineer": "Engineering", 
    "Mechanical Engineer": "Engineering",
    # ResumesCorpusDataSet.csv categories
    "Python Developer": "Python Developer",
    "Java Developer": "Java Developer",
    "Web Developer": "Web Developer", 
    "Front End Developer": "Front End Developer",
    "Software Developer": "Software Developer",
    "Database Administrator": "Database Administrator",
    "Network Administrator": "Network Administrator",
    "Systems Administrator": "Systems Administrator",
    "Security Analyst": "Security Analyst",
    "Project Manager": "Project Manager",
    # Case variations
    "hr": "HR",
    "Hr": "HR",
    "pmo": "PMO",
    "Pmo": "PMO",
}


def normalize_category(category: str) -> str:
    """
    Normalize a category string with unified rules.
    
    Consolidates logic from both category_mapper.py and dataset_cleaner.py.
    
    Steps:
    1. Strip whitespace and handle empty/invalid input
    2. Check direct mapping (exact match)
    3. Replace underscores with spaces and check mapping again
    4. Check if category is already valid in hierarchy
    5. Try title case conversion
    6. If all uppercase → Title Case
    7. Return result
    
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
    if category in CATEGORY_MAPPINGS:
        return CATEGORY_MAPPINGS[category]
        
    # Step 3: Replace underscores with spaces
    normalized = category.replace("_", " ")
    
    # Step 4: Check mapping again after underscore replacement
    if normalized in CATEGORY_MAPPINGS:
        return CATEGORY_MAPPINGS[normalized]
        
    # Step 5: Check if already valid in hierarchy
    if is_valid_category(normalized):
        return normalized
        
    # Step 6: Try title case
    title_case = normalized.title()
    if title_case in CATEGORY_MAPPINGS:
        return CATEGORY_MAPPINGS[title_case]
    if is_valid_category(title_case):
        return title_case
        
    # Step 7: If all uppercase, convert to title case
    if normalized.isupper():
        normalized = normalized.title()
        
    # Final mapping check
    if normalized in CATEGORY_MAPPINGS:
        return CATEGORY_MAPPINGS[normalized]
        
    return normalized


def get_all_unique_categories(categories: list[str]) -> list[str]:
    """
    Get all unique normalized categories from a list.
    
    Args:
        categories: List of raw category strings
        
    Returns:
        Sorted list of unique normalized categories
    """
    normalized = [normalize_category(cat) for cat in categories]
    return sorted(set(normalized))


def validate_category_input(category: str) -> str:
    """
    Validate and normalize category input.
    
    Common validation pattern extracted from multiple files.
    
    Args:
        category: Raw category input
        
    Returns:
        "Unknown" if invalid, stripped category if valid
    """
    if not category or not isinstance(category, str):
        return "Unknown"
    
    category = category.strip()
    if not category:
        return "Unknown"
        
    return category