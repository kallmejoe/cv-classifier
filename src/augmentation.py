"""Text data augmentation using statistical operations."""

import random
from typing import List


class TextAugmenter:
    """Generate synthetic training samples via token-level operations."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        random.seed(random_state)
    
    def shuffle_tokens(self, text: str, shuffle_ratio: float = 0.3) -> str:
        """Randomly shuffle tokens."""
        tokens = text.split()
        if len(tokens) < 3:
            return text
        
        n_shuffle = max(1, int(len(tokens) * shuffle_ratio))
        positions = random.sample(range(len(tokens)), min(n_shuffle, len(tokens)))
        values = [tokens[i] for i in positions]
        random.shuffle(values)
        
        for pos, val in zip(positions, values):
            tokens[pos] = val
        
        return ' '.join(tokens)
    
    def delete_random_tokens(self, text: str, delete_ratio: float = 0.1) -> str:
        """Randomly delete tokens."""
        tokens = text.split()
        if len(tokens) < 5:
            return text
        
        n_delete = max(1, int(len(tokens) * delete_ratio))
        n_delete = min(n_delete, len(tokens) // 2)
        
        keep_positions = set(range(len(tokens)))
        delete_positions = set(random.sample(range(len(tokens)), n_delete))
        keep_positions -= delete_positions
        
        tokens = [tokens[i] for i in sorted(keep_positions)]
        
        return ' '.join(tokens)
    
    def duplicate_random_tokens(self, text: str, duplicate_ratio: float = 0.1) -> str:
        """Randomly duplicate tokens."""
        tokens = text.split()
        if len(tokens) < 3:
            return text
        
        n_duplicate = max(1, int(len(tokens) * duplicate_ratio))
        duplicate_positions = random.sample(range(len(tokens)), min(n_duplicate, len(tokens)))
        
        result = []
        for i, token in enumerate(tokens):
            result.append(token)
            if i in duplicate_positions:
                result.append(token)
        
        return ' '.join(result)
    
    def random_swap(self, text: str, n_swaps: int = 3) -> str:
        """Randomly swap adjacent tokens."""
        tokens = text.split()
        if len(tokens) < 2:
            return text
        
        for _ in range(min(n_swaps, len(tokens) - 1)):
            idx = random.randint(0, len(tokens) - 2)
            tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
        
        return ' '.join(tokens)
    
    def augment(self, text: str, methods: List[str] = ['shuffle', 'delete', 'duplicate'],
                n_augmentations: int = 1) -> List[str]:
        """Generate multiple augmented samples."""
        augmented = []
        
        for _ in range(n_augmentations):
            aug_text = text
            selected_methods = random.sample(methods, k=random.randint(1, len(methods)))
            
            for method in selected_methods:
                if method == 'shuffle':
                    aug_text = self.shuffle_tokens(aug_text, shuffle_ratio=0.2)
                elif method == 'delete':
                    aug_text = self.delete_random_tokens(aug_text, delete_ratio=0.05)
                elif method == 'duplicate':
                    aug_text = self.duplicate_random_tokens(aug_text, duplicate_ratio=0.05)
                elif method == 'swap':
                    aug_text = self.random_swap(aug_text, n_swaps=2)
            
            if aug_text != text:
                augmented.append(aug_text)
        
        return augmented
    
    def augment_dataset(self, texts: List[str], labels: List[str], 
                       augmentation_factor: int = 2,
                       methods: List[str] = ['shuffle', 'delete', 'duplicate']) -> tuple:
        """Augment entire dataset."""
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        
        for text, label in zip(texts, labels):
            aug_samples = self.augment(text, methods=methods, n_augmentations=augmentation_factor)
            augmented_texts.extend(aug_samples)
            augmented_labels.extend([label] * len(aug_samples))
        
        return augmented_texts, augmented_labels
