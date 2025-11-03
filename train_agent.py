# train_agent.py
"""
Train the DQN agent on HangmanEnv using HMM predictor.
"""
import os
import time
import numpy as np
from hmm_train import HMMHybrid
from hangman_env import HangmanEnv
from dqn_agent import DQNAgent
import joblib
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def load_corpus(path="corpus.txt"):
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip().lower()
            if w and w.isalpha():
                words.append(w)
    return words

def train():
    corpus = "corpus.txt"
    assert os.path.exists(corpus), "Place corpus.txt in current directory"
    words = load_corpus(corpus)
    print("Loaded", len(words), "words")

    # build or load HMM hybrid
    hmm_path = "hmm_model.joblib"
    if os.path.exists(hmm_path):
        hmm = HMMHybrid.load(hmm_path)
        print("Loaded existing HMM model")
    else:
        hmm = HMMHybrid()
        hmm.train_from_corpus(corpus)
        hmm.save(hmm_path)
        print("Trained and saved HMM to", hmm_path)

    env = HangmanEnv(words, hmm, max_len=hmm.max_word_len, max_lives=6)
    state_dim = env._state().shape[0]
    print("State dim:", state_dim)
    agent = DQNAgent(input_dim=state_dim, lr=1e-4, gamma=0.99, tau=1e-3, hidden=512)
        # ✅ Load existing trained model if available
    if os.path.exists("models/dqn_best.pth"):
        agent.load("models/dqn_best.pth")
        print("✅ Loaded existing best model for continued training.")


    # training params
    episodes = 40000
  # increase for better performance
    batch_size = 256
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 15000.0


    best_win_rate = -1.0
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    log_interval = 200
    for ep in range(1, episodes + 1):
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1.0 * ep / epsilon_decay)
        s = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        guessed_mask = np.zeros(26, dtype=np.float32)
        while not done:
            # guessed_mask can be obtained from end of state (26 after max_len*27)
            action = agent.act(s, epsilon=epsilon, guessed_mask=guessed_mask)
            s2, r, done, info = env.step(action)
            guessed_mask = np.zeros(26, dtype=np.float32)
            # update guessed mask from s2
            state_len = len(s2)
            hmm_part = 26
            guessed_vec = s2[-(hmm_part+26):-hmm_part] if hmm_part>0 else s2[-26:]
            guessed_mask = (guessed_vec > 0.5).astype(np.float32)
            agent.push(s, action, r, s2, float(done))
            loss = agent.update(batch_size)
            s = s2
            total_reward += r
            steps += 1

        if ep % log_interval == 0:
            # quick evaluation on small sample
            wins = 0
            trials = 200
            for _ in range(trials):
                s = env.reset()
                done = False
                guessed_mask = np.zeros(26, dtype=np.float32)
                while not done:
                    action = agent.act(s, epsilon=0.0, guessed_mask=guessed_mask)
                    s, r, done, info = env.step(action)
                    guessed_vec = s[-(26+26):-26]
                    guessed_mask = (guessed_vec > 0.5).astype(np.float32)
                if 'win' in info:
                    wins += 1
            win_rate = wins / trials
            print(f"Episode {ep} eps {epsilon:.3f} eval_win_rate {win_rate:.3f}")
            # save model if improved
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                path = os.path.join(save_dir, f"dqn_best.pth")
                agent.save(path)
                print("Saved improved model to", path)

    # final save
    agent.save(os.path.join(save_dir, "dqn_final.pth"))
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    train()

