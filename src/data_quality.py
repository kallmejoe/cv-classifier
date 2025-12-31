"""
Data Quality Module - Duplicate detection and conflict resolution.

This module provides:
1. Duplicate resume detection (exact and fuzzy)
2. Category conflict detection
3. Data quality metrics and reporting
4. Automatic conflict resolution
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import hashlib


def compute_text_hash(text: str) -> str:
    """Compute MD5 hash of normalized text for fast duplicate detection."""
    # Normalize: lowercase, remove extra whitespace
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def compute_text_signature(text: str, n_shingles: int = 100) -> Set[str]:
    """
    Compute n-gram shingles for fuzzy duplicate detection.
    
    Args:
        text: Text to process
        n_shingles: Number of shingles to keep
        
    Returns:
        Set of text shingles
    """
    # Normalize
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    
    # Create 3-grams
    if len(normalized) < 3:
        return {normalized}
    
    shingles = set()
    for i in range(len(normalized) - 2):
        shingles.add(normalized[i:i+3])
    
    # Return subset if too many
    if len(shingles) > n_shingles:
        return set(list(shingles)[:n_shingles])
    
    return shingles


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


class DataQualityChecker:
    """Check and fix data quality issues in resume datasets."""
    
    def __init__(self, similarity_threshold: float = 0.9):
        """
        Initialize checker.
        
        Args:
            similarity_threshold: Threshold for fuzzy duplicate detection (0.9 = 90%)
        """
        self.similarity_threshold = similarity_threshold
        self.exact_duplicates: List[Tuple[int, int]] = []
        self.fuzzy_duplicates: List[Tuple[int, int, float]] = []
        self.category_conflicts: List[Dict] = []
        self.quality_metrics: Dict = {}
    
    def find_exact_duplicates(
        self, 
        resumes: List[str], 
        categories: List[str]
    ) -> List[Tuple[int, int]]:
        """
        Find exact duplicate resumes.
        
        Args:
            resumes: List of resume texts
            categories: Corresponding categories
            
        Returns:
            List of (index1, index2) pairs that are duplicates
        """
        hash_to_indices: Dict[str, List[int]] = defaultdict(list)
        
        for i, resume in enumerate(resumes):
            h = compute_text_hash(resume)
            hash_to_indices[h].append(i)
        
        duplicates = []
        for indices in hash_to_indices.values():
            if len(indices) > 1:
                # All combinations of duplicates
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        duplicates.append((indices[i], indices[j]))
        
        self.exact_duplicates = duplicates
        return duplicates
    
    def find_fuzzy_duplicates(
        self, 
        resumes: List[str],
        sample_size: int = 1000
    ) -> List[Tuple[int, int, float]]:
        """
        Find near-duplicate resumes using Jaccard similarity.
        
        Note: This is O(n^2) so we sample for large datasets.
        
        Args:
            resumes: List of resume texts
            sample_size: Max number of resumes to compare
            
        Returns:
            List of (index1, index2, similarity) tuples
        """
        # Compute signatures
        indices = list(range(min(len(resumes), sample_size)))
        signatures = {i: compute_text_signature(resumes[i]) for i in indices}
        
        duplicates = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                sim = jaccard_similarity(signatures[idx1], signatures[idx2])
                if sim >= self.similarity_threshold:
                    duplicates.append((idx1, idx2, sim))
        
        self.fuzzy_duplicates = duplicates
        return duplicates
    
    def find_category_conflicts(
        self,
        resumes: List[str],
        categories: List[str]
    ) -> List[Dict]:
        """
        Find resumes with conflicting category labels.
        
        Same resume text with different categories = conflict.
        
        Args:
            resumes: List of resume texts
            categories: Corresponding categories
            
        Returns:
            List of conflict dictionaries
        """
        # Group by text hash
        hash_to_data: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
        
        for i, (resume, category) in enumerate(zip(resumes, categories)):
            h = compute_text_hash(resume)
            hash_to_data[h].append((i, category, resume[:100]))
        
        conflicts = []
        for h, entries in hash_to_data.items():
            categories_seen = set(cat for _, cat, _ in entries)
            if len(categories_seen) > 1:
                conflicts.append({
                    'hash': h,
                    'indices': [idx for idx, _, _ in entries],
                    'categories': list(categories_seen),
                    'sample': entries[0][2],  # First 100 chars
                    'count': len(entries)
                })
        
        self.category_conflicts = conflicts
        return conflicts
    
    def compute_quality_metrics(
        self,
        resumes: List[str],
        categories: List[str]
    ) -> Dict:
        """
        Compute comprehensive data quality metrics.
        
        Args:
            resumes: List of resume texts
            categories: Corresponding categories
            
        Returns:
            Dictionary of quality metrics
        """
        # Basic counts
        total = len(resumes)
        unique_resumes = len(set(compute_text_hash(r) for r in resumes))
        unique_categories = len(set(categories))
        
        # Length statistics
        lengths = [len(r) for r in resumes]
        avg_length = sum(lengths) / total if total > 0 else 0
        min_length = min(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0
        
        # Short resume count (< 100 chars)
        short_resumes = sum(1 for l in lengths if l < 100)
        
        # Empty resume count
        empty_resumes = sum(1 for r in resumes if not r or not r.strip())
        
        # Category distribution
        category_counts = Counter(categories)
        min_samples = min(category_counts.values()) if category_counts else 0
        max_samples = max(category_counts.values()) if category_counts else 0
        
        # Imbalance ratio
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        # Find duplicates and conflicts
        self.find_exact_duplicates(resumes, categories)
        self.find_category_conflicts(resumes, categories)
        
        metrics = {
            'total_samples': total,
            'unique_resumes': unique_resumes,
            'duplicate_resumes': total - unique_resumes,
            'duplicate_percentage': 100 * (total - unique_resumes) / total if total > 0 else 0,
            'unique_categories': unique_categories,
            'avg_length': avg_length,
            'min_length': min_length,
            'max_length': max_length,
            'short_resumes': short_resumes,
            'empty_resumes': empty_resumes,
            'exact_duplicate_pairs': len(self.exact_duplicates),
            'category_conflicts': len(self.category_conflicts),
            'min_samples_per_category': min_samples,
            'max_samples_per_category': max_samples,
            'imbalance_ratio': imbalance_ratio,
        }
        
        self.quality_metrics = metrics
        return metrics
    
    def resolve_duplicates(
        self,
        resumes: List[str],
        categories: List[str],
        strategy: str = 'keep_first'
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Remove duplicate resumes.
        
        Args:
            resumes: List of resume texts
            categories: Corresponding categories
            strategy: 'keep_first' (only option now - keep_most_specific removed)
            
        Returns:
            Tuple of (cleaned_resumes, cleaned_categories, kept_indices)
        """
        seen_hashes: Dict[str, int] = {}
        kept_indices = []
        
        for i, (resume, category) in enumerate(zip(resumes, categories)):
            h = compute_text_hash(resume)
            
            if h not in seen_hashes:
                seen_hashes[h] = i
                kept_indices.append(i)
            # For duplicates, just keep first occurrence (strategy parameter ignored)
        
        cleaned_resumes = [resumes[i] for i in kept_indices]
        cleaned_categories = [categories[i] for i in kept_indices]
        
        return cleaned_resumes, cleaned_categories, kept_indices
    
    def print_quality_report(self) -> None:
        """Print a formatted data quality report."""
        if not self.quality_metrics:
            print("No quality metrics computed. Run compute_quality_metrics() first.")
            return
        
        m = self.quality_metrics
        
        print("\n" + "="*70)
        print("DATA QUALITY REPORT")
        print("="*70)
        
        # Sample statistics
        print("\n📊 SAMPLE STATISTICS:")
        print(f"   Total samples:        {m['total_samples']:,}")
        print(f"   Unique resumes:       {m['unique_resumes']:,}")
        print(f"   Duplicates:           {m['duplicate_resumes']:,} ({m['duplicate_percentage']:.1f}%)")
        print(f"   Unique categories:    {m['unique_categories']}")
        
        # Resume quality
        print("\n📝 RESUME QUALITY:")
        print(f"   Average length:       {m['avg_length']:,.0f} chars")
        print(f"   Min length:           {m['min_length']:,} chars")
        print(f"   Max length:           {m['max_length']:,} chars")
        print(f"   Short resumes (<100): {m['short_resumes']:,}")
        print(f"   Empty resumes:        {m['empty_resumes']:,}")
        
        # Category balance
        print("\n⚖️  CATEGORY BALANCE:")
        print(f"   Min samples/category: {m['min_samples_per_category']:,}")
        print(f"   Max samples/category: {m['max_samples_per_category']:,}")
        print(f"   Imbalance ratio:      {m['imbalance_ratio']:.1f}x")
        
        # Issues
        print("\n⚠️  ISSUES DETECTED:")
        print(f"   Exact duplicate pairs: {m['exact_duplicate_pairs']:,}")
        print(f"   Category conflicts:    {m['category_conflicts']:,}")
        
        # Category conflicts details
        if self.category_conflicts:
            print("\n🔀 CATEGORY CONFLICTS (top 5):")
            for conflict in self.category_conflicts[:5]:
                cats = ", ".join(conflict['categories'])
                print(f"   [{conflict['count']} samples] {cats}")
                print(f"      Sample: \"{conflict['sample']}...\"")
        
        # Quality score
        score = 100
        if m['duplicate_percentage'] > 10:
            score -= 20
        if m['category_conflicts'] > 0:
            score -= 10 * min(m['category_conflicts'], 5)
        if m['short_resumes'] > m['total_samples'] * 0.05:
            score -= 10
        if m['imbalance_ratio'] > 10:
            score -= 15
        
        score = max(0, score)
        
        print(f"\n📈 OVERALL QUALITY SCORE: {score}/100")
        
        if score >= 80:
            print("   ✅ Good quality dataset")
        elif score >= 60:
            print("   ⚠️  Some issues need attention")
        else:
            print("   ❌ Significant quality issues - cleaning recommended")
        
        print("="*70)


def clean_dataset(
    resumes: List[str],
    categories: List[str],
    normalize_categories: bool = True,
    remove_duplicates: bool = True,
    remove_short: bool = True,
    min_length: int = 100,
    resolve_conflicts: bool = True,
    verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Clean a dataset by removing issues.
    
    Args:
        resumes: List of resume texts
        categories: Corresponding categories
        normalize_categories: Apply category normalization
        remove_duplicates: Remove duplicate resumes
        remove_short: Remove short resumes
        min_length: Minimum resume length
        resolve_conflicts: Resolve category conflicts (keep most specific)
        verbose: Print progress
        
    Returns:
        Tuple of (cleaned_resumes, cleaned_categories)
    """
    from .category_mapper import map_category
    
    if verbose:
        print("\n🧹 Cleaning dataset...")
        print(f"   Initial size: {len(resumes)} samples")
    
    cleaned_resumes = list(resumes)
    cleaned_categories = list(categories)
    
    # Step 1: Remove empty/short resumes
    if remove_short:
        valid = [(r, c) for r, c in zip(cleaned_resumes, cleaned_categories) 
                 if r and len(r.strip()) >= min_length]
        if valid:
            cleaned_resumes, cleaned_categories = zip(*valid)  # type: ignore
            cleaned_resumes = list(cleaned_resumes)
            cleaned_categories = list(cleaned_categories)
        if verbose:
            removed = len(resumes) - len(cleaned_resumes)
            print(f"   Removed {removed} short/empty resumes")
    
    # Step 2: Normalize categories
    if normalize_categories:
        cleaned_categories = [
            map_category(cat, resume) 
            for cat, resume in zip(cleaned_categories, cleaned_resumes)
        ]
        if verbose:
            print(f"   Normalized {len(set(categories))} → {len(set(cleaned_categories))} categories")
    
    # Step 3: Remove duplicates (keep most specific)
    if remove_duplicates:
        checker = DataQualityChecker()
        strategy = 'keep_most_specific' if resolve_conflicts else 'keep_first'
        cleaned_resumes, cleaned_categories, _ = checker.resolve_duplicates(
            cleaned_resumes, cleaned_categories, strategy=strategy
        )
        if verbose:
            print(f"   Removed duplicates → {len(cleaned_resumes)} samples")
    
    if verbose:
        print(f"   Final size: {len(cleaned_resumes)} samples")
    
    return cleaned_resumes, cleaned_categories
