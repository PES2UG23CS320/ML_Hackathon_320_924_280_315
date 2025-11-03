# hangman_env.py
"""
Hangman environment for RL training.
State: vector combining masked word one-hot (fixed max length), guessed letters binary vector, and HMM prob vector.
"""
import random
import numpy as np
from collections import namedtuple
from hmm_train import HMMHybrid, ALPHABET, IDX

class HangmanEnv:
    def __init__(self, words, hmm_model: HMMHybrid, max_len=None, max_lives=6):
        self.words = words
        self.hmm = hmm_model
        self.max_lives = max_lives
        self.max_len = max_len or self.hmm.max_word_len
        # sanity cap
        self.max_len = max(self.max_len, max(len(w) for w in words))
        self.reset()

    def sample_word(self):
        return random.choice(self.words)

    def reset(self, word=None):
        self.word = word or self.sample_word()
        self.word = self.word.lower()
        self.mask = ['_' for _ in self.word]
        self.guessed = set()
        self.wrong = 0
        return self._state()

    def allowed_actions(self):
        return [i for i in range(26) if ALPHABET[i] not in self.guessed]

    def step(self, action_idx):
        # action_idx: integer 0..25
        ch = ALPHABET[action_idx]
        reward = 0.0
        done = False
        info = {}
        if ch in self.guessed:
            reward += -1.0  # repeated guess penalty
            info['repeat'] = True
        else:
            self.guessed.add(ch)
            if ch in self.word:
                 reward += 6.0  # instead of 5.0 # correct guess reward
            else:
                self.wrong += 1
                reward += -2.0  # wrong guess penalty

        if '_' not in self.mask:
            done = True
            reward += 10.0  # win reward
            info['win'] = True
        elif self.wrong >= self.max_lives:
            done = True
            reward += -10.0  # loss penalty
            info['loss'] = True

        return self._state(), reward, done, info

    def _state(self):
        # masked word padded to max_len
        masked = ''.join(self.mask)
        # encode masked_word: for each pos a 27-dim one-hot (26 letters + blank)
        mw = np.zeros((self.max_len, 27), dtype=np.float32)
        for i in range(self.max_len):
            if i < len(masked):
                c = masked[i]
                if c == '_' or c == '?':
                    mw[i, 26] = 1.0
                else:
                    mw[i, IDX[c]] = 1.0
            else:
                mw[i, 26] = 1.0  # padded blank
        mw = mw.flatten()  # max_len * 27

        guessed_vec = np.zeros(26, dtype=np.float32)
        for g in self.guessed:
            if g in IDX:
                guessed_vec[IDX[g]] = 1.0

        hmm_probs = self.hmm.get_hmm_letter_probs(masked, self.guessed).astype(np.float32)

        # final state vector
        state = np.concatenate([mw, guessed_vec, hmm_probs])
        return state

    def render(self):
        print("Word:", ''.join(self.mask), "Wrong:", self.wrong, "Guessed:", sorted(list(self.guessed)))

