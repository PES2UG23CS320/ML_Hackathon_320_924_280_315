# play.py
"""
Interactive test script for Hangman DQN agent.
You can input your own word, and the trained agent will try to guess it.
Shows each step: guessed letter, masked word, reward, lives remaining.
"""

import numpy as np
import os
from hmm_train import HMMHybrid
from dqn_agent import DQNAgent
from hangman_env import ALPHABET
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_masked_word(word, masked, guess):
    """Reveals guessed letters in the masked word."""
    updated = list(masked)
    for i, c in enumerate(word):
        if c == guess:
            updated[i] = c
    return ''.join(updated)

def play_game(agent, hmm, word, max_lives=6):
    """Simulate a single hangman game using trained agent."""
    word = word.lower()
    masked_word = "_" * len(word)
    guessed = set()
    lives = max_lives
    total_reward = 0.0
    print(f"\n🎮 Starting game with word of length {len(word)}")
    print(f"Masked: {masked_word}\n")

    while lives > 0 and "_" in masked_word:
        # Build guessed vector
        guessed_vec = np.zeros(26, dtype=np.float32)
        for g in guessed:
            guessed_vec[ord(g) - 97] = 1.0

        # Get HMM probabilities
        hmm_probs = hmm.get_hmm_letter_probs(masked_word, guessed)

        # Build state (only what agent needs)
        # We skip the one-hot word encoding for simplicity (agent expects full state)
        # So we just feed dummy one-hot padding of zeros and append guessed_vec + hmm_probs
        # Build state padded to same length as during training
        # Build state padded to same length as during training
    max_len = hmm.max_word_len
    mw = np.zeros(max_len * 27, dtype=np.float32)
    for i, c in enumerate(masked_word):
        if i < max_len:
            if c == '_' or c == '?':
                mw[i * 27 + 26] = 1.0  # blank symbol
        else:
            mw[i * 27 + (ord(c) - 97)] = 1.0

state = np.concatenate([mw, guessed_vec, hmm_probs])

state = np.concatenate([mw, guessed_vec, hmm_probs])


        # Predict next action (best letter)
        action = agent.act(state, epsilon=0.0, guessed_mask=guessed_vec)
        guess = ALPHABET[action]

        # Determine reward
        if guess in guessed:
            reward = -1.0  # repeated guess
            print(f"Letter '{guess}' already guessed. Reward {reward:+}")
        elif guess in word:
            masked_word = update_masked_word(word, masked_word, guess)
            reward = +5.0
            print(f"✅ Guess: '{guess}' | Masked: {masked_word} | Reward {reward:+} | Lives: {lives}")
        else:
            lives -= 1
            reward = -2.0
            print(f"❌ Guess: '{guess}' | Masked: {masked_word} | Reward {reward:+} | Lives: {lives}")

        guessed.add(guess)
        total_reward += reward

        if "_" not in masked_word:
            print("\n🎉 Agent won! The word was:", word)
            print(f"Total reward: {total_reward:+}")
            return

    print("\n💀 Agent lost! The word was:", word)
    print(f"Total reward: {total_reward:+}")

def main():
    # Load trained models
    if not os.path.exists("hmm_model.joblib"):
        raise SystemExit("❌ hmm_model.joblib not found. Train it first with hmm_train.py.")
    if not os.path.exists("models/dqn_best.pth"):
        raise SystemExit("❌ models/dqn_best.pth not found. Train it first with train_agent.py.")

    print("✅ Loading trained models...")
    hmm = HMMHybrid.load("hmm_model.joblib")

    # Dummy input dim (approx same as in train_agent)
    input_dim = hmm.max_word_len * 27 + 26 + 26
    agent = DQNAgent(input_dim=input_dim, hidden=512)
    agent.load("models/dqn_best.pth")

    # Ask user for custom word
    word = input("Enter a word for the agent to guess (letters only): ").strip().lower()
    if not word.isalpha():
        print("❌ Please enter a valid word (letters only).")
        return

    play_game(agent, hmm, word, max_lives=6)

if __name__ == "__main__":
    main()
