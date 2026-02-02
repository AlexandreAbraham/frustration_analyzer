#!/usr/bin/env python3
"""
Frustration Analyzer for ChatGPT Conversations
Measures frustration indicators (ALL CAPS, insults, negative sentiment) in user messages over time.
Uses ML-based profanity detection and sentiment analysis.
"""

import json
import re
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ML-based detection
from profanity_check import predict_prob as profanity_prob  # from alt-profanity-check
from textblob import TextBlob

# French profanity list (profanity_check is English-focused, so we supplement)
FRENCH_PROFANITY = {
    "gogol", "merde", "putain", "bordel", "con", "connard", "connasse", "conne",
    "crétin", "crétine", "imbécile", "débile", "chiant", "chiante", "chier",
    "foutre", "foutue", "foutu", "enculé", "salaud", "salope", "ta gueule",
    "nique", "niquer", "pute", "encule", "batard", "bâtard", "couille",
    "branleur", "branleuse", "pétasse", "abruti", "abrutie",
}

def load_conversations(filepath):
    """Load conversations from ChatGPT export JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_user_messages(conversations):
    """
    Extract all user messages with their timestamps.
    Returns list of (timestamp, message_text) tuples.
    """
    messages = []

    for conv in conversations:
        mapping = conv.get('mapping', {})

        for node_id, node in mapping.items():
            message = node.get('message')
            if not message:
                continue

            author = message.get('author', {})
            if author.get('role') != 'user':
                continue

            content = message.get('content', {})
            parts = content.get('parts', [])

            # Get timestamp
            create_time = message.get('create_time')
            if not create_time:
                create_time = conv.get('create_time')

            if not create_time:
                continue

            # Combine all text parts
            text = ' '.join(str(p) for p in parts if isinstance(p, str))
            if text.strip():
                messages.append((create_time, text))

    return messages

# Pre-compiled regex patterns for performance
_CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')
_INLINE_CODE_RE = re.compile(r'`[^`]+`')
_URL_RE = re.compile(r'https?://\S+')
_SPECIAL_CHARS_RE = re.compile(r'[{}\[\]();=<>|&^%$#@!]')
_CODE_INDICATORS = frozenset([
    '=>', '->', '::', '==', '!=', '<=', '>=', '&&', '||',
    '});', ');', '};', '{}', '[]', '()',
    'import ', 'export ', 'function ', 'def ', 'class ',
    'const ', 'let ', 'var ', 'return ', 'if (', 'for (', 'while (',
])

def remove_code_and_logs(text):
    """Remove code blocks, logs, and technical content from text."""
    # Quick check: if text is short and has no code indicators, return as-is
    if len(text) < 200 and '```' not in text and '`' not in text:
        return text

    # Remove markdown code blocks
    text = _CODE_BLOCK_RE.sub(' ', text)

    # Remove inline code
    text = _INLINE_CODE_RE.sub(' ', text)

    # Remove URLs
    text = _URL_RE.sub(' ', text)

    # Fast line filtering
    lines = text.split('\n')
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip lines with code indicators
        if any(ind in stripped for ind in _CODE_INDICATORS):
            continue

        # Skip lines with too many special characters (>15%)
        if len(stripped) > 10:
            special_count = len(_SPECIAL_CHARS_RE.findall(stripped))
            if special_count / len(stripped) > 0.15:
                continue

        filtered_lines.append(line)

    return '\n'.join(filtered_lines)

def count_caps_words(text):
    """Count words that are ALL CAPS (3+ letters)."""
    words = re.findall(r'\b[A-Z]{3,}\b', text)
    # Filter out common acronyms/technical terms that aren't frustration indicators
    technical_terms = {
        # General tech
        'API', 'URL', 'HTML', 'CSS', 'JSON', 'HTTP', 'HTTPS', 'SQL', 'PDF', 'XML',
        'USA', 'UK', 'EU', 'USD', 'EUR', 'CEO', 'CTO', 'CFO',
        # Programming
        'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS',
        'NULL', 'TRUE', 'FALSE', 'NONE', 'NIL', 'NAN', 'INF',
        'INT', 'STR', 'BOOL', 'FLOAT', 'CHAR', 'VOID', 'CONST',
        'EOF', 'EOL', 'EOS', 'ASCII', 'UTF', 'UUID', 'GUID',
        'RAM', 'CPU', 'GPU', 'SSD', 'HDD', 'USB', 'HDMI',
        'AWS', 'GCP', 'CLI', 'GUI', 'IDE', 'SDK', 'JDK', 'JVM',
        'DOM', 'CSS', 'SVG', 'PNG', 'JPG', 'GIF', 'RGB', 'HEX',
        'SSH', 'FTP', 'TCP', 'UDP', 'DNS', 'TLS', 'SSL', 'VPN',
        'TODO', 'FIXME', 'NOTE', 'XXX', 'HACK',
        'README', 'LICENSE', 'CHANGELOG',
        # Log levels (in case some slip through)
        'DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'FATAL', 'TRACE',
        # Common abbreviations
        'ASAP', 'FAQ', 'RSVP', 'ETA', 'FYI', 'TBD', 'TBA', 'AKA',
        'NASA', 'NATO', 'UNESCO',
    }
    caps_words = [w for w in words if w not in technical_terms]
    return len(caps_words)

def count_french_profanity(text):
    """Count French profanity words."""
    text_lower = text.lower()
    count = 0
    for word in FRENCH_PROFANITY:
        if ' ' in word:
            count += text_lower.count(word)
        else:
            count += len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
    return count

def analyze_frustration_ml(text):
    """
    Analyze frustration using ML models.
    Returns a dict with individual scores:
    - profanity: ML-based profanity probability (English)
    - french: French profanity score
    - sentiment: Negative sentiment score
    """
    result = {'profanity': 0.0, 'french': 0.0, 'sentiment': 0.0}

    if not text.strip():
        return result

    # Split into sentences for better analysis
    sentences = re.split(r'[.!?\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return result

    # 1. ML-based profanity detection (works best on English)
    try:
        profanity_scores = profanity_prob(sentences)
        result['profanity'] = max(profanity_scores) if len(profanity_scores) > 0 else 0
    except:
        pass

    # 2. French profanity (word-based since ML model is English)
    french_count = count_french_profanity(text)
    word_count = len(re.findall(r'\b\w+\b', text))
    result['french'] = min(french_count / max(word_count, 1) * 10, 1.0)  # Scale up, cap at 1

    # 3. Sentiment analysis (negative = frustrated)
    try:
        blob = TextBlob(text)
        # Polarity is -1 to 1, we want negative sentiment
        result['sentiment'] = max(0, -blob.sentiment.polarity)  # Convert to 0-1 frustration
    except:
        pass

    return result

def count_total_words(text):
    """Count total words in text."""
    return len(re.findall(r'\b\w+\b', text))

def get_week_key(timestamp):
    """Convert Unix timestamp to ISO week key (YYYY-WW)."""
    try:
        # Handle timestamps that might be in milliseconds
        if timestamp > 1e12:
            timestamp = timestamp / 1000

        dt = datetime.fromtimestamp(timestamp)

        # Sanity check: skip dates too far in the past or future
        if dt.year < 2020 or dt.year > 2030:
            return None

        iso_cal = dt.isocalendar()
        return f"{iso_cal[0]}-W{iso_cal[1]:02d}"
    except (ValueError, OSError):
        return None

def get_week_start_date(week_key):
    """Convert week key back to a date for plotting."""
    year, week = week_key.split('-W')
    return datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")

def analyze_frustration(messages):
    """
    Analyze frustration metrics per week using ML models.
    Returns dict with weekly stats broken down by source.
    """
    weekly_stats = defaultdict(lambda: {
        'total_messages': 0,
        'total_words': 0,
        'caps_words': 0,
        'profanity_score': 0.0,  # ML profanity (English)
        'french_score': 0.0,     # French profanity
        'sentiment_score': 0.0,  # Negative sentiment
    })

    skipped = 0
    total = len(messages)
    for i, (timestamp, text) in enumerate(messages):
        if i % 500 == 0:
            print(f"  Processing message {i}/{total}...")

        week = get_week_key(timestamp)
        if week is None:
            skipped += 1
            continue

        # Filter out code and logs before analysis
        clean_text = remove_code_and_logs(text)

        stats = weekly_stats[week]
        ml_scores = analyze_frustration_ml(clean_text)

        stats['total_messages'] += 1
        stats['total_words'] += count_total_words(clean_text)
        stats['caps_words'] += count_caps_words(clean_text)
        stats['profanity_score'] += ml_scores['profanity']
        stats['french_score'] += ml_scores['french']
        stats['sentiment_score'] += ml_scores['sentiment']

    if skipped > 0:
        print(f"  (Skipped {skipped} messages with invalid timestamps)")

    return weekly_stats

def calculate_rates(weekly_stats):
    """Calculate frustration rates per week for each source."""
    rates = {}
    for week, stats in weekly_stats.items():
        total_words = stats['total_words']
        total_messages = stats['total_messages']
        if total_words > 0 and total_messages > 0:
            # Caps rate as percentage of words
            caps_rate = (stats['caps_words'] / total_words) * 100
            # Individual ML scores as average per message (0-100 scale)
            profanity_rate = (stats['profanity_score'] / total_messages) * 100
            french_rate = (stats['french_score'] / total_messages) * 100
            sentiment_rate = (stats['sentiment_score'] / total_messages) * 100
            # Combined
            combined_rate = (caps_rate * 0.25 + profanity_rate * 0.25 +
                           french_rate * 0.25 + sentiment_rate * 0.25)

            rates[week] = {
                'caps_rate': caps_rate,
                'profanity_rate': profanity_rate,
                'french_rate': french_rate,
                'sentiment_rate': sentiment_rate,
                'combined_rate': combined_rate,
                'total_messages': total_messages,
                'total_words': total_words,
            }
    return rates

def smooth_data(data, window=4):
    """Apply moving average smoothing."""
    import numpy as np
    weights = np.ones(window) / window
    # Pad edges to keep same length
    padded = np.pad(data, (window//2, window-1-window//2), mode='edge')
    return np.convolve(padded, weights, mode='valid')

# Major ChatGPT/OpenAI model releases
MODEL_RELEASES = [
    (datetime(2022, 11, 30), "ChatGPT\nLaunch"),
    (datetime(2023, 3, 14), "GPT-4"),
    (datetime(2023, 11, 6), "GPT-4\nTurbo"),
    (datetime(2024, 5, 13), "GPT-4o"),
    (datetime(2024, 9, 12), "o1"),
    (datetime(2025, 2, 27), "GPT-4.5"),
    (datetime(2025, 8, 8), "GPT-5"),
]

def plot_frustration(rates, output_file='frustration_plot.png'):
    """Generate two plots: 4 sources (subplots) + combined."""
    import numpy as np

    # Sort by week
    sorted_weeks = sorted(rates.keys())

    if not sorted_weeks:
        print("No data to plot!")
        return

    dates = [get_week_start_date(w) for w in sorted_weeks]
    date_min, date_max = min(dates), max(dates)

    # Extract each source
    caps_rates = [rates[w]['caps_rate'] for w in sorted_weeks]
    profanity_rates = [rates[w]['profanity_rate'] for w in sorted_weeks]
    french_rates = [rates[w]['french_rate'] for w in sorted_weeks]
    sentiment_rates = [rates[w]['sentiment_rate'] for w in sorted_weeks]
    combined_rates = [rates[w]['combined_rate'] for w in sorted_weeks]

    # === PLOT 1: 4 separate subplots ===
    fig1, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig1.suptitle('Frustration Analysis by Source', fontsize=14, fontweight='bold')

    plots = [
        (axes[0], caps_rates, 'ALL CAPS', 'orange', 'Shouting (ALL CAPS words)'),
        (axes[1], profanity_rates, 'ML Profanity (EN)', 'red', 'ML-detected profanity (English)'),
        (axes[2], french_rates, 'French Profanity', 'blue', 'French swear words'),
        (axes[3], sentiment_rates, 'Negative Sentiment', 'purple', 'Negative sentiment (TextBlob)'),
    ]

    for ax, data, ylabel, color, title in plots:
        smoothed = smooth_data(np.array(data), window=4)
        ax.fill_between(dates, data, alpha=0.2, color=color)
        ax.plot(dates, data, color=color, linewidth=0.8, alpha=0.4)
        ax.plot(dates, smoothed, color=color, linewidth=2)

        for release_date, model_name in MODEL_RELEASES:
            if date_min <= release_date <= date_max:
                ax.axvline(x=release_date, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_ylabel(f'{ylabel} (%)', fontsize=10)
        ax.set_title(title, fontsize=11, loc='left')
        ax.grid(True, alpha=0.3)

    # Model labels on top plot
    for release_date, model_name in MODEL_RELEASES:
        if date_min <= release_date <= date_max:
            axes[0].annotate(model_name.replace('\n', ' '), xy=(release_date, max(caps_rates) * 0.9),
                           xytext=(3, 0), textcoords='offset points',
                           fontsize=8, color='#555', fontweight='bold', ha='left', va='top', rotation=90)

    axes[-1].set_xlabel('Date', fontsize=12)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('frustration_sources.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to: frustration_sources.png")

    # === PLOT 2: Combined/Overall ===
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    fig2.suptitle('Overall Frustration Index', fontsize=14, fontweight='bold')

    smoothed_combined = smooth_data(np.array(combined_rates), window=4)
    ax2.fill_between(dates, combined_rates, alpha=0.2, color='purple')
    ax2.plot(dates, combined_rates, color='purple', linewidth=0.8, alpha=0.4, label='Raw')
    ax2.plot(dates, smoothed_combined, color='purple', linewidth=2.5, label='Smoothed (4-week)')

    for release_date, model_name in MODEL_RELEASES:
        if date_min <= release_date <= date_max:
            ax2.axvline(x=release_date, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
            ax2.annotate(model_name.replace('\n', ' '), xy=(release_date, max(combined_rates) * 0.95),
                        xytext=(3, 0), textcoords='offset points',
                        fontsize=9, color='#c0392b', fontweight='bold', ha='left', va='top')

    ax2.set_ylabel('Combined Frustration Rate (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('frustration_combined.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to: frustration_combined.png")

    plt.show()

def print_summary(rates):
    """Print summary statistics."""
    if not rates:
        print("No data to summarize!")
        return

    sorted_weeks = sorted(rates.keys())

    # Find peaks for each source
    max_caps = max(rates.items(), key=lambda x: x[1]['caps_rate'])
    max_profanity = max(rates.items(), key=lambda x: x[1]['profanity_rate'])
    max_french = max(rates.items(), key=lambda x: x[1]['french_rate'])
    max_sentiment = max(rates.items(), key=lambda x: x[1]['sentiment_rate'])

    total_messages = sum(r['total_messages'] for r in rates.values())
    total_words = sum(r['total_words'] for r in rates.values())

    print("\n" + "="*60)
    print("FRUSTRATION ANALYSIS SUMMARY (by source)")
    print("="*60)
    print(f"Date range: {sorted_weeks[0]} to {sorted_weeks[-1]}")
    print(f"Total weeks analyzed: {len(rates)}")
    print(f"Total messages: {total_messages:,}")
    print(f"Total words: {total_words:,}")
    print(f"\nPeak ALL CAPS: {max_caps[0]} ({max_caps[1]['caps_rate']:.2f}%)")
    print(f"Peak ML Profanity (EN): {max_profanity[0]} ({max_profanity[1]['profanity_rate']:.2f}%)")
    print(f"Peak French Profanity: {max_french[0]} ({max_french[1]['french_rate']:.2f}%)")
    print(f"Peak Negative Sentiment: {max_sentiment[0]} ({max_sentiment[1]['sentiment_rate']:.2f}%)")
    print("="*60)

def main():
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else 'conversations.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'frustration_plot.png'

    print(f"Loading conversations from: {filepath}")
    conversations = load_conversations(filepath)
    print(f"Found {len(conversations)} conversations")

    print("Extracting user messages...")
    messages = extract_user_messages(conversations)
    print(f"Found {len(messages)} user messages")

    print("Analyzing frustration levels...")
    weekly_stats = analyze_frustration(messages)
    rates = calculate_rates(weekly_stats)

    print_summary(rates)
    plot_frustration(rates, output_file)

if __name__ == '__main__':
    main()
