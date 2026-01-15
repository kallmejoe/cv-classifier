"""
Dataset Cleaner - Fixes mislabeled categories and prepares clean training data.

This module:
1. Identifies and removes mislabeled samples (AUTOMOBILE, BPO issues)
2. Normalizes category names to match hierarchy
3. Validates categories against the hierarchy tree
4. Removes tiny categories that can't be learned well
"""

import pandas as pd
import re
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

from src.category_hierarchy import get_all_leaf_categories, is_valid_category
from src.utils.logging_utils import get_logger

logger = get_logger()


# =============================================================================
# MISLABELED CATEGORY DETECTION
# =============================================================================

# Keywords that indicate actual job roles for detecting mislabeling
JOB_ROLE_KEYWORDS: Dict[str, List[str]] = {
    "automobile_actual": [
        "automobile",
        "automotive",
        "mechanic",
        "vehicle",
        "car technician",
        "auto repair",
        "auto body",
        "car dealer",
        "truck driver",
        "transporter",
    ],
    "bpo_actual": [
        "bpo",
        "business process",
        "call center",
        "customer service outsourcing",
        "bpo operations",
        "bpo manager",
        "contact center",
    ],
    "it_software": [
        "software",
        "developer",
        "programmer",
        "engineer",
        "python",
        "java",
        "javascript",
        "react",
        "angular",
        "node",
        "database",
        "sql",
        "api",
        "backend",
        "frontend",
        "devops",
        "cloud",
        "aws",
        "azure",
    ],
    "data_analytics": [
        "data scientist",
        "data analyst",
        "machine learning",
        "etl",
        "datastage",
        "informatica",
        "data warehouse",
        "analytics",
        "bi developer",
    ],
    "finance_insurance": [
        "claims",
        "insurance",
        "underwriter",
        "actuary",
        "policy",
        "adjuster",
        "usaa",
        "financial analyst",
        "accountant",
    ],
    "legal": ["attorney", "lawyer", "legal", "paralegal", "law firm"],
    "hr_recruiting": [
        "recruiter",
        "hr ",
        "human resources",
        "talent acquisition",
        "staffing",
    ],
    "admin_office": [
        "administrative",
        "secretary",
        "office manager",
        "coordinator",
        "executive assistant",
        "receptionist",
    ],
}


@dataclass
class CleaningStats:
    """Statistics from cleaning operation."""

    original_count: int
    removed_mislabeled: int
    removed_tiny_categories: int
    removed_duplicates: int
    final_count: int
    categories_before: int
    categories_after: int
    removed_samples: List[Tuple[int, str, str]]  # (index, category, reason)


def detect_mislabel(resume_text: str, labeled_category: str) -> Optional[str]:
    """
    Detect if a resume is mislabeled.

    Args:
        resume_text: Resume content
        labeled_category: The assigned category label

    Returns:
        Suggested correct category if mislabeled, None if label seems correct
    """
    text_lower = resume_text.lower()

    # Check AUTOMOBILE category for mislabeling
    if labeled_category.upper() == "AUTOMOBILE":
        # Count actual automobile-related keywords
        auto_score = sum(
            1 for kw in JOB_ROLE_KEYWORDS["automobile_actual"] if kw in text_lower
        )

        # Count IT/software keywords
        it_score = sum(1 for kw in JOB_ROLE_KEYWORDS["it_software"] if kw in text_lower)
        data_score = sum(
            1 for kw in JOB_ROLE_KEYWORDS["data_analytics"] if kw in text_lower
        )
        finance_score = sum(
            1 for kw in JOB_ROLE_KEYWORDS["finance_insurance"] if kw in text_lower
        )
        legal_score = sum(1 for kw in JOB_ROLE_KEYWORDS["legal"] if kw in text_lower)
        hr_score = sum(
            1 for kw in JOB_ROLE_KEYWORDS["hr_recruiting"] if kw in text_lower
        )
        admin_score = sum(
            1 for kw in JOB_ROLE_KEYWORDS["admin_office"] if kw in text_lower
        )

        # If more IT/other keywords than auto keywords, it's mislabeled
        max_other = max(
            it_score, data_score, finance_score, legal_score, hr_score, admin_score
        )

        if auto_score < 2 and max_other > auto_score:
            if it_score == max_other:
                return "Information-Technology"
            elif data_score == max_other:
                return "Data Science"
            elif finance_score == max_other:
                return "Finance"
            elif legal_score == max_other:
                return "Advocate"
            elif hr_score == max_other:
                return "HR"
            elif admin_score == max_other:
                return "Operations Manager"

        # USAA employees are often mislabeled (USAA = insurance company)
        if "usaa" in text_lower or "united services automobile" in text_lower:
            if (
                finance_score >= 2
                or "claims" in text_lower
                or "insurance" in text_lower
            ):
                return "Finance"

    # Check BPO category for mislabeling
    if labeled_category.upper() == "BPO":
        bpo_score = sum(1 for kw in JOB_ROLE_KEYWORDS["bpo_actual"] if kw in text_lower)
        it_score = sum(1 for kw in JOB_ROLE_KEYWORDS["it_software"] if kw in text_lower)

        if bpo_score < 2 and it_score > bpo_score:
            return "Information-Technology"

    return None


# =============================================================================
# CATEGORY NORMALIZATION
# =============================================================================

# Simple normalization: uppercase dataset categories to our hierarchy
CATEGORY_NORMALIZATION: Dict[str, str] = {
    # Resume.csv UPPERCASE → our hierarchy names
    "ACCOUNTANT": "Accountant",
    "ADVOCATE": "Advocate",
    "AGRICULTURE": "Agriculture",
    "APPAREL": "Apparel",
    "ARTS": "Arts",
    "AUTOMOBILE": "Automobile",  # Will be handled specially (mislabeled removal)
    "AVIATION": "Aviation",
    "BANKING": "Banking",
    "BPO": "Bpo",  # Will be handled specially (mislabeled removal)
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
    # ResumesCorpusDataSet.csv categories (already match hierarchy exactly)
    # Listed here for explicitness and future-proofing
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


# =============================================================================
# MAIN CLEANING FUNCTION
# =============================================================================


def clean_dataset(
    df: pd.DataFrame,
    resume_column: str = "Resume",
    category_column: str = "Category",
    remove_mislabeled: bool = True,
    remove_tiny_categories: bool = True,
    min_samples_per_category: int = 20,
    remove_duplicates: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, CleaningStats]:
    """
    Clean the dataset by fixing mislabeling and normalizing categories.

    Args:
        df: Input DataFrame
        resume_column: Name of resume text column
        category_column: Name of category column
        remove_mislabeled: Whether to remove detected mislabeled samples
        remove_tiny_categories: Whether to remove categories with few samples
        min_samples_per_category: Minimum samples to keep a category
        remove_duplicates: Whether to remove duplicate resumes
        verbose: Print progress

    Returns:
        Tuple of (cleaned_df, stats)
    """
    if verbose:
        logger.info("\n" + "=" * 70)
        logger.info("DATASET CLEANING")
        logger.info("=" * 70)

    # Make copy to avoid modifying original
    df = df.copy()
    original_count = len(df)
    original_categories = df[category_column].nunique()
    removed_samples: List[Tuple[int, str, str]] = []

    if verbose:
        logger.info(f"\nOriginal: {original_count} samples, {original_categories} categories")

    # Step 1: Normalize category names
    if verbose:
        logger.info("\nStep 1: Normalizing category names...")

    from .utils.category_utils import normalize_category
    
    df["normalized_category"] = df[category_column].apply(normalize_category)

    # Step 2: Detect and remove mislabeled samples
    removed_mislabeled = 0
    if remove_mislabeled:
        if verbose:
            logger.info("Step 2: Detecting mislabeled samples...")

        indices_to_remove = []

        for idx, row in df.iterrows():
            suggested = detect_mislabel(row[resume_column], row[category_column])
            if suggested is not None:
                # This sample is mislabeled - remove it
                indices_to_remove.append(idx)
                removed_samples.append(
                    (idx, row[category_column], f"mislabeled, suggested: {suggested}")
                )

        if indices_to_remove:
            df = df.drop(indices_to_remove)
            removed_mislabeled = len(indices_to_remove)
            if verbose:
                logger.info(f"  - Removed {removed_mislabeled} mislabeled samples")

    # Step 3: Remove tiny categories
    removed_tiny = 0
    if remove_tiny_categories:
        if verbose:
            logger.info(
                f"Step 3: Removing categories with < {min_samples_per_category} samples..."
            )

        category_counts = df["normalized_category"].value_counts()
        tiny_categories = category_counts[
            category_counts < min_samples_per_category
        ].index.tolist()

        if tiny_categories:
            before = len(df)
            tiny_mask = df["normalized_category"].isin(tiny_categories)

            for idx, row in df[tiny_mask].iterrows():
                removed_samples.append(
                    (idx, row["normalized_category"], "tiny category")
                )

            df = df[~tiny_mask]
            removed_tiny = before - len(df)

            if verbose:
                logger.info(
                    f"  - Removed {removed_tiny} samples from tiny categories: {tiny_categories}"
                )

    # Step 4: Remove duplicates
    removed_dups = 0
    if remove_duplicates:
        if verbose:
            logger.info("Step 4: Removing duplicate resumes...")

        before = len(df)
        df = df.drop_duplicates(subset=resume_column, keep="first")
        removed_dups = before - len(df)

        if verbose:
            logger.info(f"  - Removed {removed_dups} duplicate resumes")

    # Step 5: Validate against hierarchy
    if verbose:
        logger.info("Step 5: Validating categories against hierarchy...")

    valid_categories = set(get_all_leaf_categories())
    df_categories = set(df["normalized_category"].unique())

    unknown = df_categories - valid_categories
    if unknown:
        if verbose:
            logger.info(f"  - Warning: Categories not in hierarchy: {unknown}")

    # Final stats
    final_count = len(df)
    final_categories = df["normalized_category"].nunique()

    stats = CleaningStats(
        original_count=original_count,
        removed_mislabeled=removed_mislabeled,
        removed_tiny_categories=removed_tiny,
        removed_duplicates=removed_dups,
        final_count=final_count,
        categories_before=original_categories,
        categories_after=final_categories,
        removed_samples=removed_samples,
    )

    if verbose:
        logger.info(f"\n" + "=" * 70)
        logger.info("CLEANING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Original samples:       {stats.original_count}")
        logger.info(f"  Removed mislabeled:     {stats.removed_mislabeled}")
        logger.info(f"  Removed tiny categories:{stats.removed_tiny_categories}")
        logger.info(f"  Removed duplicates:     {stats.removed_duplicates}")
        logger.info(f"  Final samples:          {stats.final_count}")
        logger.info(f"  Categories: {stats.categories_before} → {stats.categories_after}")
        logger.info("=" * 70)

    return df, stats


def get_clean_dataset(
    include_corpus: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """
    Load and clean the combined dataset.

    Args:
        include_corpus: Whether to include ResumesCorpusDataSet.csv (30K tech samples)
        verbose: Print progress information

    Returns:
        Cleaned DataFrame with normalized categories
    """
    from src.data_loader import load_combined_datasets

    # Load raw data - now includes corpus by default for tech coverage
    df = load_combined_datasets(
        include_resume=True, include_updated=True, include_corpus=include_corpus
    )

    # Standardize column name
    if "Resume_str" in df.columns:
        df = df.rename(columns={"Resume_str": "Resume"})

    # Clean it
    df_clean, stats = clean_dataset(
        df, resume_column="Resume", category_column="Category", verbose=verbose
    )

    # Rename normalized category to Category
    df_clean["Category"] = df_clean["normalized_category"]
    df_clean = df_clean.drop(columns=["normalized_category"])

    return df_clean
