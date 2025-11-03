"""
Train a compact character-level HMM-like model on corpus.txt
and expose a function:
    get_hmm_letter_probs(masked_word, guessed_set)
→ returns 26-dim probability vector for A–Z

This hybrid HMM uses:
 - letter bigram frequencies
 - emission identity probabilities
 - pattern-matching over corpus for contextual guesses

Saved model: hmm_model.joblib
"""

import re
import numpy as np
import joblib
from collections import Counter
import os
import math

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
IDX = {c: i for i, c in enumerate(ALPHABET)}

class HMMHybrid:
    def __init__(self):
        # Transition and emission counts
        self.trans_counts = np.ones((26, 26))  # bigram smoothing
        self.emit_counts = np.ones((26, 26))   # identity emissions
        self.unigram = np.ones(26)
        self.words = []
        self.max_word_len = 0

    def train_from_corpus(self, corpus_path):
        """Train bigram/unigram distributions from corpus"""
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip().lower()
                if not w or not re.fullmatch(r"[a-z]+", w):
                    continue
                self.words.append(w)
                self.max_word_len = max(self.max_word_len, len(w))
                prev = None
                for ch in w:
                    i = IDX[ch]
                    self.unigram[i] += 1
                    if prev is not None:
                        self.trans_counts[IDX[prev], i] += 1
                    # self.emit_counts models likelihood of emission
                    self.emit_counts[i, i] += 1
                    prev = ch

        # Normalize to probabilities
        self.trans_probs = self.trans_counts / self.trans_counts.sum(axis=1, keepdims=True)
        self.emit_probs = self.emit_counts / self.emit_counts.sum(axis=1, keepdims=True)
        self.unigram_probs = self.unigram / np.sum(self.unigram)
        print(f"✅ Trained HMMHybrid on {len(self.words)} words (max length {self.max_word_len})")

    def save(self, path="hmm_model.joblib"):
        joblib.dump(self, path)
        print(f"💾 Saved HMM model to {path}")

    @staticmethod
    def load(path="hmm_model.joblib"):
        return joblib.load(path)

    def get_hmm_letter_probs(self, masked_word, guessed_set):
        """
        Improved version:
         - Safely handles zero-match cases (no divide warnings)
         - Combines pattern matching with fallback unigram probabilities
        """
        masked = masked_word.lower()
        L = len(masked)
        match_counts = np.zeros(26)
        total_matches = 0

        # pattern: replace blanks with '.'
        pattern = ''.join('.' if c == '_' or c == '?' else c for c in masked)
        pat = re.compile('^' + pattern + '$')

        for w in self.words:
            if len(w) != L:
                continue
            if not pat.match(w):
                continue
            total_matches += 1
            for pos, ch in enumerate(w):
                if masked[pos] == '_' or masked[pos] == '?':
                    match_counts[IDX[ch]] += 1

        # Case 1: found matching words
        if total_matches > 0 and np.sum(match_counts) > 0:
            probs = match_counts / np.sum(match_counts)
        else:
            # Case 2: fallback to unigram distribution
            probs = self.unigram_probs.copy()

        # Remove guessed letters
        for g in guessed_set:
            if g in IDX:
                probs[IDX[g]] = 0.0

        # Normalize again
        s = probs.sum()
        if s <= 0:
            probs = np.ones(26)
            for g in guessed_set:
                if g in IDX:
                    probs[IDX[g]] = 0.0
            probs /= probs.sum()
        else:
            probs /= s

        return probs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="corpus.txt", help="Path to corpus.txt")
    parser.add_argument("--out", default="hmm_model.joblib", help="Output model file")
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        raise SystemExit("❌ corpus.txt not found in current directory.")

    model = HMMHybrid()
    print(f"📘 Training HMM Hybrid on {args.corpus} ...")
    model.train_from_corpus(args.corpus)
    model.save(args.out)
    print("✅ HMM training complete.")
