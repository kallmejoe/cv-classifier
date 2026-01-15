"""
Category Hierarchy Module - Defines the category tree structure.

This module provides:
1. Hierarchical category tree (root → domain → specialization → specific)
2. Path lookup for any category
3. Backpropagation logic to get parent categories
4. Confidence-based tier selection

Tree Structure:
    Root (Level 0) - e.g., "Professional"
    └── Domain (Level 1) - e.g., "Technology"
        └── Specialization (Level 2) - e.g., "Software Development"
            └── Specific Role (Level 3) - e.g., "Python Developer"
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# CATEGORY HIERARCHY TREE
# =============================================================================
# Structure: Domain → Specialization → [Specific Roles]
# Each leaf category maps to its full path in the tree

CATEGORY_TREE: Dict[str, Dict[str, List[str]]] = {
    "Technology": {
        "Software Development": [
            "Python Developer",
            "Java Developer", 
            "DotNet Developer",
            "Software Developer",
            "Blockchain",
        ],
        "Web Development": [
            "Web Developer",
            "Front End Developer",
            "Designer",
        ],
        "Data & Analytics": [
            "Data Science",
            "ETL Developer",
            "Hadoop",
            "Database",
            "Database Administrator",
        ],
        "Infrastructure & DevOps": [
            "DevOps Engineer",
            "Network Security Engineer",
            "Network Administrator",
            "Systems Administrator",
            "Security Analyst",
            "SAP Developer",
        ],
        "Quality Assurance": [
            "Testing",
            "Automation Testing",
        ],
        "General IT": [
            "Information-Technology",
        ],
        "Project Management": [
            "Project Manager",
        ],
    },
    "Engineering": {
        "Engineering": [
            "Engineering",  # General engineering category
        ],
    },
    "Business & Management": {
        "Business Analysis": [
            "Business Analyst",
            "Consultant",
        ],
        "Management": [
            "Operations Manager",
            "PMO",
            "Business-Development",
        ],
        "Sales & Marketing": [
            "Sales",
            "Public-Relations",
            "Digital-Media",
        ],
    },
    "Finance & Legal": {
        "Finance": [
            "Finance",
            "Banking",
            "Accountant",
        ],
        "Legal": [
            "Advocate",
        ],
    },
    "Healthcare & Services": {
        "Healthcare": [
            "Healthcare",
            "Fitness",
        ],
        "Human Resources": [
            "HR",
        ],
        "Hospitality": [
            "Chef",
        ],
        "Arts & Design": [
            "Arts",
        ],
    },
    "Specialized Industries": {
        "Education": [
            "Teacher",
        ],
        "Construction": [
            "Construction",
        ],
        "Agriculture": [
            "Agriculture",
        ],
        "Aviation": [
            "Aviation",
        ],
        "Retail": [
            "Apparel",
        ],
    },
}

# Root category for very low confidence predictions
ROOT_CATEGORY = "Professional"


# =============================================================================
# BUILD REVERSE LOOKUP MAPS
# =============================================================================

def _build_category_paths() -> Dict[str, List[str]]:
    """
    Build a lookup dictionary mapping each leaf category to its full path.
    
    Returns:
        Dict mapping category name to path list
        e.g., "Python Developer" → ["Technology", "Software Development", "Python Developer"]
    """
    paths: Dict[str, List[str]] = {}
    
    for domain, specializations in CATEGORY_TREE.items():
        for specialization, categories in specializations.items():
            for category in categories:
                # Path: [Domain, Specialization, Category]
                paths[category] = [domain, specialization, category]
    
    return paths


def _build_parent_lookup() -> Dict[str, str]:
    """
    Build a lookup dictionary mapping each category to its parent.
    
    Returns:
        Dict mapping category to parent
        e.g., "Python Developer" → "Software Development"
    """
    parents: Dict[str, str] = {}
    
    for domain, specializations in CATEGORY_TREE.items():
        for specialization, categories in specializations.items():
            # Specialization's parent is Domain
            parents[specialization] = domain
            # Category's parent is Specialization
            for category in categories:
                parents[category] = specialization
    
    return parents


# Pre-built lookup tables
CATEGORY_PATHS: Dict[str, List[str]] = _build_category_paths()
PARENT_LOOKUP: Dict[str, str] = _build_parent_lookup()


# =============================================================================
# PUBLIC API
# =============================================================================

def get_category_path(category: str) -> List[str]:
    """
    Get the full hierarchy path for a category.
    
    Args:
        category: The leaf category name
        
    Returns:
        List of categories from domain to specific
        e.g., ["Technology", "Software Development", "Python Developer"]
        
    If category not found, returns [ROOT_CATEGORY, category]
    """
    if category in CATEGORY_PATHS:
        return CATEGORY_PATHS[category].copy()
    
    # Unknown category - return minimal path
    return [ROOT_CATEGORY, category]


def get_parent(category: str) -> Optional[str]:
    """
    Get the parent category.
    
    Args:
        category: Category name
        
    Returns:
        Parent category name, or None if no parent
    """
    return PARENT_LOOKUP.get(category)


def get_domain(category: str) -> str:
    """
    Get the top-level domain for a category.
    
    Args:
        category: Any category name
        
    Returns:
        Domain name (Level 1), or ROOT_CATEGORY if not found
    """
    path = get_category_path(category)
    return path[0] if path else ROOT_CATEGORY


def get_specialization(category: str) -> Optional[str]:
    """
    Get the specialization (Level 2) for a category.
    
    Args:
        category: Leaf category name
        
    Returns:
        Specialization name, or None if not found
    """
    path = get_category_path(category)
    return path[1] if len(path) >= 2 else None


def backpropagate(category: str, levels: int = 1) -> str:
    """
    Move up the hierarchy by specified number of levels.
    
    Args:
        category: Starting category
        levels: Number of levels to go up (1 = parent, 2 = grandparent)
        
    Returns:
        Ancestor category at specified level
    """
    path = get_category_path(category)
    
    if not path:
        return ROOT_CATEGORY
    
    # Calculate target index (going backwards from end)
    target_idx = max(0, len(path) - 1 - levels)
    return path[target_idx]


def get_all_leaf_categories() -> List[str]:
    """
    Get all leaf (most specific) categories.
    
    Returns:
        List of all leaf category names
    """
    return list(CATEGORY_PATHS.keys())


def get_all_domains() -> List[str]:
    """
    Get all top-level domains.
    
    Returns:
        List of domain names
    """
    return list(CATEGORY_TREE.keys())


def is_valid_category(category: str) -> bool:
    """
    Check if a category exists in the hierarchy.
    
    Args:
        category: Category name to check
        
    Returns:
        True if category is a valid leaf category
    """
    return category in CATEGORY_PATHS


def find_consensus_domain(categories: List[str]) -> str:
    """
    Find the most common domain among a list of categories.
    
    Useful for low-confidence predictions where we look at top-N predictions
    and find the most likely domain.
    
    Args:
        categories: List of category names
        
    Returns:
        Most common domain, or ROOT_CATEGORY if no consensus
    """
    if not categories:
        return ROOT_CATEGORY
    
    domains = [get_domain(cat) for cat in categories]
    
    # Count domain occurrences
    domain_counts: Dict[str, int] = {}
    for domain in domains:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    # Find most common
    if domain_counts:
        return max(domain_counts.keys(), key=lambda d: domain_counts[d])
    
    return ROOT_CATEGORY


# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================

@dataclass
class ConfidenceThresholds:
    """Thresholds for confidence-based tier selection."""
    high: float = 0.7      # Return specific category
    medium: float = 0.4    # Return specialization (parent)
    # Below medium: Return domain (grandparent) or use consensus


DEFAULT_THRESHOLDS = ConfidenceThresholds()


@dataclass
class HierarchicalPrediction:
    """Result of a hierarchical prediction."""
    category: str                      # The category to return (may be parent/domain based on confidence)
    specific_category: str             # The model's original prediction
    confidence: float                  # Model confidence
    tier: str                          # "high", "medium", or "low"
    path: List[str]                    # Full hierarchy path
    requires_review: bool = False      # True if low confidence
    top_suggestions: Optional[List[str]] = None  # Top-N for low confidence cases


def select_by_confidence(
    predicted_category: str,
    confidence: float,
    top_predictions: Optional[List[str]] = None,
    thresholds: ConfidenceThresholds = DEFAULT_THRESHOLDS
) -> HierarchicalPrediction:
    """
    Select appropriate category level based on confidence.
    
    Args:
        predicted_category: Model's top prediction
        confidence: Prediction confidence (0-1)
        top_predictions: Optional list of top-N predictions for low confidence
        thresholds: Confidence thresholds for tier selection
        
    Returns:
        HierarchicalPrediction with appropriate category level
    """
    path = get_category_path(predicted_category)
    
    # Tier 1: High confidence - return specific category
    if confidence >= thresholds.high:
        return HierarchicalPrediction(
            category=predicted_category,
            specific_category=predicted_category,
            confidence=confidence,
            tier="high",
            path=path,
            requires_review=False
        )
    
    # Tier 2: Medium confidence - return specialization (one level up)
    elif confidence >= thresholds.medium:
        parent = backpropagate(predicted_category, levels=1)
        return HierarchicalPrediction(
            category=parent,
            specific_category=predicted_category,
            confidence=confidence,
            tier="medium",
            path=path,
            requires_review=False
        )
    
    # Tier 3: Low confidence - return domain or use consensus
    else:
        if top_predictions:
            # Use consensus from top predictions
            consensus_domain = find_consensus_domain(top_predictions)
            category = consensus_domain
        else:
            # Just go up two levels to domain
            category = backpropagate(predicted_category, levels=2)
        
        return HierarchicalPrediction(
            category=category,
            specific_category=predicted_category,
            confidence=confidence,
            tier="low",
            path=path,
            requires_review=True,
            top_suggestions=top_predictions
        )
