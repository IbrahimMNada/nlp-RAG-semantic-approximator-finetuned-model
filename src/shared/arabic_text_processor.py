"""Arabic text preprocessing service."""
import re


# ── Inline replacements for camel-tools utilities ──

# Arabic diacritic (tashkeel) Unicode range
_DIACRITICS_RE = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')


def dediac_ar(text: str) -> str:
    """Remove Arabic diacritics (tashkeel)."""
    return _DIACRITICS_RE.sub('', text)


def normalize_alef_ar(text: str) -> str:
    """Normalize Alef variants (أ إ آ ٱ) to bare Alef (ا)."""
    return re.sub(r'[\u0622\u0623\u0625\u0671]', '\u0627', text)


def normalize_alef_maksura_ar(text: str) -> str:
    """Normalize Alef Maksura (ى) to Ya (ي)."""
    return text.replace('\u0649', '\u064A')


def normalize_teh_marbuta_ar(text: str) -> str:
    """Normalize Teh Marbuta (ة) to Ha (ه)."""
    return text.replace('\u0629', '\u0647')


def simple_word_tokenize(text: str) -> list:
    """Split Arabic text on whitespace (equivalent to camel-tools simple_word_tokenize)."""
    return text.split()


# Arabic stopwords - comprehensive list from CAMeL Tools
ARABIC_STOPWORDS = {
    # Prepositions
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'حتى', 'منذ', 'خلال', 'بين',
    'تحت', 'فوق', 'أمام', 'خلف', 'حول', 'دون', 'سوى', 'عند', 'لدى',

    # Conjunctions & particles
    'و', 'ف', 'ثم', 'أو', 'أم', 'بل', 'لكن', 'لأن', 'إذ', 'إذا', 'إذن',
    'كما', 'كذلك', 'حيث', 'إلا', 'ألا', 'أما',

    # Negation & emphasis
    'لا', 'لم', 'لن', 'ما', 'ليس', 'إن', 'أن', 'قد',

    # Pronouns (separate forms)
    'هو', 'هي', 'هم', 'هن', 'أنت', 'أنتم', 'أنتن', 'نحن', 'أنا',
    'إياه', 'إياها', 'إياهم',

    # Attached pronoun equivalents (as tokens)
    'له', 'لها', 'لهم', 'لهن',
    'به', 'بها', 'بهم',
    'منه', 'منها', 'منهم',
    'عليه', 'عليها', 'عليهم',
    'فيه', 'فيها', 'فيهم',
    'إليه', 'إليها', 'إليهم',

    # Demonstratives
    'هذا', 'هذه', 'هؤلاء', 'ذلك', 'تلك', 'أولئك',

    # Relative pronouns
    'الذي', 'التي', 'الذين', 'اللاتي', 'اللواتي',

    # Interrogatives
    'هل', 'كيف', 'متى', 'أين', 'لماذا', 'ماذا', 'كم', 'أي',

    # Quantifiers
    'كل', 'بعض', 'كثير', 'قليل', 'أكثر', 'أقل',

    # Temporal words
    'الآن', 'اليوم', 'أمس', 'غدا', 'حين', 'وقت', 'سابقا', 'لاحقا',

    # Modal / auxiliary verbs
    'كان', 'كانت', 'يكون', 'تكون', 'ليس', 'ليست',

    # Common discourse fillers
    'أيضا', 'فقط', 'أيضا', 'عادة', 'غالبا', 'تقريبا',

    # Definite article & prefixes (⚠️ optional – see note below)
    'ال', 'بال', 'وال', 'فال', 'كال', 'لل'
}


# Pattern to match reference markers like [1], [٣], [ ٣ ], etc.
# Matches: [ optional spaces ] [ Arabic numerals ٠-٩ or Western numerals 0-9 ] [ optional spaces ]
REFERENCE_PATTERN = re.compile(r'\[\s*[٠-٩0-9]+\s*\]')


class ArabicTextProcessor:
    """Service for processing Arabic text using CAMeL Tools."""
    
    def __init__(self, stopwords=None):
        """
        Initialize Arabic text processor.
        
        Args:
            stopwords: Optional custom set of stopwords. If None, uses default ARABIC_STOPWORDS.
        """
        self.stopwords = stopwords if stopwords is not None else ARABIC_STOPWORDS
    
    @staticmethod
    def pre_clean(text: str) -> str:
        """
        Clean text by removing standalone punctuation and normalizing.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Remove standalone punctuation tokens
        text = re.sub(r'\s+[.,:؛،]\s+', ' ', text)
        
        # Normalize repeated punctuation
        text = re.sub(r'[.,:؛،]{2,}', lambda m: m.group(0)[0], text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def remove_references(text: str) -> str:
        """
        Remove reference markers like [1], [٣], [ ٣ ].
        
        Args:
            text: Input text
            
        Returns:
            Text with reference markers removed
        """
        if not text:
            return text
        return REFERENCE_PATTERN.sub('', text)
    
    @staticmethod
    def remove_tashkeel(text: str) -> str:
        """
        Remove Arabic diacritics (Tashkeel) using CAMeL Tools.
        
        Args:
            text: Input text
            
        Returns:
            Text without diacritics
        """
        if not text:
            return text
        return dediac_ar(text)
    
    @staticmethod
    def normalize_arabic(text: str) -> str:
        """
        Normalize Arabic characters using CAMeL Tools.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return text
        # Normalize different forms of Alef
        text = normalize_alef_ar(text)
        # Normalize Alef Maksura to Ya
        text = normalize_alef_maksura_ar(text)
        # Normalize Teh Marbuta
        text = normalize_teh_marbuta_ar(text)
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove Arabic stop words from text using CAMeL Tools tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        if not text:
            return text
        # Use CAMeL Tools tokenizer for better Arabic word segmentation
        words = simple_word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def preprocess_arabic_text(self, text: str) -> str:
        """
        Apply all Arabic preprocessing steps using CAMeL Tools.
        
        This method applies the following transformations in order:
        1. Pre-clean: remove standalone punctuation and normalize whitespace
        2. Remove reference markers [1], [٣], etc.
        3. Remove Tashkeel (diacritics)
        4. Normalize Arabic characters
        5. Remove stop words
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return text
        # Pre-clean: remove standalone punctuation and normalize
        text = self.pre_clean(text)
        # Remove reference markers [1], [٣], etc.
        text = self.remove_references(text)
        # Remove Tashkeel (diacritics)
        text = self.remove_tashkeel(text)
        # Normalize Arabic characters
        text = self.normalize_arabic(text)
        # Remove stop words
        text = self.remove_stopwords(text)
        return text
    
    @staticmethod
    def tokenize_arabic(text: str) -> list:
        """
        Tokenize Arabic text using CAMeL Tools word tokenizer.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        return simple_word_tokenize(text)
    
    def get_stopwords_count(self) -> int:
        """
        Get the number of stopwords in the current stopwords set.
        
        Returns:
            Number of stopwords
        """
        return len(self.stopwords)
