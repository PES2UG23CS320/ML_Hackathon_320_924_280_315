# play.py
"""
Interactive test script for Hangman DQN + HMM hybrid agent.
Now shows top-5 letter predictions with realistic probabilities (0–1 range).
"""

import numpy as np
import os
import torch
from hmm_train import HMMHybrid
from dqn_agent import DQNAgent
from hangman_env import ALPHABET

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------- Utility Functions ------------------- #

def update_masked_word(word, masked, guess):
    """Reveals guessed letters in the masked word."""
    updated = list(masked)
    for i, c in enumerate(word):
        if c == guess:
            updated[i] = c
    return ''.join(updated)


def softmax(x, temperature=1.0):
    """Numerically stable softmax with optional temperature scaling."""
    x = np.array(x, dtype=np.float32)
    x = x / max(temperature, 1e-8)       # control sharpness
    x = x - np.max(x)                    # stability shift
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x + 1e-8)


# ------------------- Game Logic ------------------- #

def play_game(agent, hmm, word, max_lives=6):
    """Simulate a single Hangman game using trained agent."""
    word = word.lower()
    masked_word = "_" * len(word)
    guessed = set()
    lives = max_lives
    total_reward = 0.0
    correct_guesses = 0
    wrong_guesses = 0

    print(f"\n🎮 Starting game with word of length {len(word)}")
    print(f"Masked: {masked_word}\n")

    while lives > 0 and "_" in masked_word:
        # Build guessed vector
        guessed_vec = np.zeros(26, dtype=np.float32)
        for g in guessed:
            guessed_vec[ord(g) - 97] = 1.0

        # Get HMM probabilities
        hmm_probs = hmm.get_hmm_letter_probs(masked_word, guessed)

        # --- Build state padded to same size as training ---
        max_len = hmm.max_word_len
        mw = np.zeros(max_len * 27, dtype=np.float32)
        for i, c in enumerate(masked_word):
            if i < max_len:
                if c == "_" or c == "?":
                    mw[i * 27 + 26] = 1.0  # blank symbol
                else:
                    mw[i * 27 + (ord(c) - 97)] = 1.0

        # Final state (same dim as training)
        state = np.concatenate([mw, guessed_vec, hmm_probs])

        # ---- SHOW TOP 5 PREDICTED LETTERS WITH SOFTMAX PROBABILITIES ---- #
        agent.net.eval()
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            q_values = agent.net(s).cpu().numpy().ravel()

            # Mask out already guessed letters
            q_values = q_values - 1e6 * guessed_vec

            # Compute softmax probabilities (realistic confidence)
            probs = softmax(q_values, temperature=0.8)

            # Pick top 5 letters
            top5_idx = np.argsort(probs)[-5:][::-1]
            print("💭 Top 5 letter predictions (probability 0–1):")
            for idx in top5_idx:
                print(f"   {ALPHABET[idx]} → {probs[idx]:.3f}")
            print()

        # Predict next action (best letter)
        action = int(np.argmax(probs))
        guess = ALPHABET[action]

        # Determine reward and update
        if guess in guessed:
            reward = -1.0  # repeated guess
            print(f"⚠ Letter '{guess}' already guessed. Reward {reward:+}")
        elif guess in word:
            masked_word = update_masked_word(word, masked_word, guess)
            reward = +5.0
            correct_guesses += 1
            print(f"✅ Guess: '{guess}' | Masked: {masked_word} | Reward {reward:+} | Lives: {lives}")
        else:
            lives -= 1
            reward = -2.0
            wrong_guesses += 1
            print(f"❌ Guess: '{guess}' | Masked: {masked_word} | Reward {reward:+} | Lives: {lives}")

        guessed.add(guess)
        total_reward += reward

        # Check win
        if "_" not in masked_word:
            print("\n🎉 Agent won! The word was:", word)
            print(f"✅ Total correct guesses: {correct_guesses}")
            print(f"❌ Total wrong guesses: {wrong_guesses}")
            print(f"🏆 Total reward: {total_reward:+.1f}")
            return

    # Loss case
    print("\n💀 Agent lost! The word was:", word)
    print(f"✅ Total correct guesses: {correct_guesses}")
    print(f"❌ Total wrong guesses: {wrong_guesses}")
    print(f"🏆 Total reward: {total_reward:+.1f}")


# ------------------- Main ------------------- #

def main():
    # Load trained models
    if not os.path.exists("hmm_model.joblib"):
        raise SystemExit("❌ hmm_model.joblib not found. Train it first with hmm_train.py.")
    if not os.path.exists("models/dqn_best.pth"):
        raise SystemExit("❌ models/dqn_best.pth not found. Train it first with train_agent.py.")

    print("✅ Loading trained models...")
    hmm = HMMHybrid.load("hmm_model.joblib")

    # Input dimension (same as training)
    input_dim = hmm.max_word_len * 27 + 26 + 26
    agent = DQNAgent(input_dim=input_dim, hidden=512)
    agent.load("models/dqn_best.pth")

    # Ask user for a custom word
    word = input("Enter a word for the agent to guess (letters only): ").strip().lower()
    if not word.isalpha():
        print("❌ Please enter a valid word (letters only).")
        return

    play_game(agent, hmm, word, max_lives=6)


if _name_ == "_main_":
    main()
