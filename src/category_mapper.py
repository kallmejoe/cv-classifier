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
