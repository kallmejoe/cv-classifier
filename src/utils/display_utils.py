"""
Display utilities for consistent formatting and progress reporting.

This module provides utilities to replace the 26+ repeated decorative
printing patterns found throughout the codebase.
"""

from typing import Dict, Any, Optional


def print_header(title: str, width: int = 70, char: str = "=") -> None:
    """
    Print a formatted section header with consistent styling.
    
    Replaces repeated patterns like:
    print("=" * 70)
    print("SECTION HEADER")
    print("=" * 70)
    
    Args:
        title: Header title to display
        width: Width of the header line
        char: Character to use for header line
    """
    print(f"\n{char * width}")
    print(title.center(width))
    print(char * width)


def print_subheader(title: str, width: int = 70, char: str = "-") -> None:
    """
    Print a formatted subheader.
    
    Args:
        title: Subheader title to display
        width: Width of the header line
        char: Character to use for subheader line
    """
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def print_step(step_num: int, total_steps: int, description: str, width: int = 70) -> None:
    """
    Print a formatted step indicator.
    
    Replaces patterns like:
    print(f"[1/5] Tuning LinearSVC...")
    
    Args:
        step_num: Current step number (1-based)
        total_steps: Total number of steps
        description: Step description
        width: Width for formatting
    """
    step_header = f"[{step_num}/{total_steps}] {description}"
    print(f"\n{step_header}")
    if len(step_header) < width:
        print("-" * len(step_header))


def print_distribution_table(
    data: Dict[str, int], 
    title: str = "Distribution",
    max_items: int = 20,
    bar_width: int = 30,
    name_width: int = 35
) -> None:
    """
    Print a formatted distribution table with ASCII bars.
    
    Replaces complex loops in category_mapper.py and other files.
    
    Args:
        data: Dictionary of item -> count
        title: Table title
        max_items: Maximum items to display
        bar_width: Maximum width of ASCII bars
        name_width: Width for name column
    """
    print(f"\n{title} ({len(data)} categories):")
    
    # Sort by count (descending) and limit items
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:max_items]
    
    if not sorted_items:
        print("  No data to display")
        return
        
    max_count = max(count for _, count in sorted_items)
    
    for name, count in sorted_items:
        # Create ASCII bar
        bar_length = min(int(count * bar_width / max_count), bar_width) if max_count > 0 else 0
        bar = "█" * bar_length
        
        # Format and print row
        name_formatted = name[:name_width].ljust(name_width)
        print(f"  {name_formatted} {count:5d} {bar}")
    
    if len(data) > max_items:
        print(f"  ... and {len(data) - max_items} more categories")


def print_stats_summary(stats: Dict[str, Any], title: str = "Summary") -> None:
    """
    Print a formatted statistics summary.
    
    Args:
        stats: Dictionary of statistic name -> value
        title: Summary title
    """
    print(f"\n{title}:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.2f}")
        elif isinstance(value, int):
            print(f"  {key:25s}: {value:,}")
        else:
            print(f"  {key:25s}: {value}")


def print_progress_update(current: int, total: int, description: str = "", width: int = 50) -> None:
    """
    Print a progress bar update.
    
    Args:
        current: Current progress value
        total: Total expected value
        description: Optional description
        width: Width of progress bar
    """
    if total == 0:
        percentage = 0
    else:
        percentage = min(100, int(100 * current / total))
    
    filled_width = int(width * current / total) if total > 0 else 0
    bar = "█" * filled_width + "░" * (width - filled_width)
    
    desc_text = f" {description}" if description else ""
    print(f"\rProgress: |{bar}| {percentage:3d}% ({current:,}/{total:,}){desc_text}", end="", flush=True)


def print_file_saved(file_path: str, description: str = "File") -> None:
    """
    Standard file save notification.
    
    Replaces repeated patterns in evaluation.py and other files.
    
    Args:
        file_path: Path where file was saved
        description: Description of what was saved
    """
    print(f"{description} saved to: {file_path}")


def print_data_quality_status(
    total_samples: int,
    duplicates: int = 0, 
    short_samples: int = 0,
    categories: int = 0,
    use_emojis: bool = True
) -> None:
    """
    Print standardized data quality status.
    
    Args:
        total_samples: Total number of samples
        duplicates: Number of duplicate samples
        short_samples: Number of short/empty samples
        categories: Number of unique categories
        use_emojis: Whether to use emoji indicators
    """
    status_good = "✅" if use_emojis else "[OK]"
    status_warn = "⚠️ " if use_emojis else "[WARN]"
    status_bad = "❌" if use_emojis else "[ERROR]"
    
    print(f"\n📊 Data Quality Summary:" if use_emojis else "\nData Quality Summary:")
    print(f"  Total samples: {total_samples:,}")
    
    if duplicates > 0:
        dup_pct = 100 * duplicates / total_samples
        icon = status_warn if dup_pct < 20 else status_bad
        print(f"  {icon} Duplicates: {duplicates:,} ({dup_pct:.1f}%)")
    
    if short_samples > 0:
        short_pct = 100 * short_samples / total_samples
        icon = status_warn if short_pct < 5 else status_bad
        print(f"  {icon} Short samples: {short_samples:,} ({short_pct:.1f}%)")
    
    if categories > 0:
        print(f"  Categories: {categories:,} unique")
    
    # Overall assessment
    issues = duplicates + short_samples
    if issues == 0:
        print(f"  {status_good} Dataset quality: Good")
    elif issues < total_samples * 0.1:
        print(f"  {status_warn} Dataset quality: Acceptable (minor issues)")
    else:
        print(f"  {status_bad} Dataset quality: Poor (significant issues)")


def format_time_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_section_separator(width: int = 70) -> None:
    """Print a simple section separator."""
    print("\n" + "-" * width)