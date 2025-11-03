# evaluate_agent.py
"""
Evaluate a saved DQN + HMM model on N games and print the metrics + final score.
"""
import os
import numpy as np
from hmm_train import HMMHybrid
from hangman_env import HangmanEnv
from dqn_agent import DQNAgent
import random

def load_corpus(path="corpus.txt"):
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip().lower()
            if w and w.isalpha():
                words.append(w)
    return words

def evaluate(model_path="models/dqn_best.pth", games=2000):
    words = load_corpus("corpus.txt")
    hmm = HMMHybrid.load("hmm_model.joblib")
    env = HangmanEnv(words, hmm, max_len=hmm.max_word_len, max_lives=6)
    agent = DQNAgent(input_dim=env._state().shape[0])
    agent.load(model_path)
    wins = 0
    total_wrong = 0
    total_repeats = 0

    for i in range(games):
        s = env.reset()
        done = False
        while not done:
            # build guessed mask
            guessed_vec = s[-(26+26):-26]  # guessed vector
            guessed_mask = (guessed_vec > 0.5).astype(np.float32)
            a = agent.act(s, epsilon=0.0, guessed_mask=guessed_mask)
            prev_guessed = set(env.guessed)
            s2, r, done, info = env.step(a)
            # track metrics
            if 'repeat' in info:
                total_repeats += 1
            if 'loss' in info:
                total_wrong += env.wrong
            if 'win' in info:
                wins += 1
                total_wrong += env.wrong
            s = s2

    success_rate = wins / games
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeats * 2)
    print("Games:", games)
    print("Wins:", wins)
    print("Success rate:", success_rate)
    print("Total wrong guesses:", total_wrong)
    print("Total repeated guesses:", total_repeats)
    print("Final score:", final_score)
    return {
        "games": games,
        "wins": wins,
        "success_rate": success_rate,
        "total_wrong": total_wrong,
        "total_repeats": total_repeats,
        "final_score": final_score
    }

if __name__ == "__main__":
    res = evaluate()

