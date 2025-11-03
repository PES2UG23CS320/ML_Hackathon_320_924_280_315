# ML_Hackathon_320_924_280_315
We built a hybrid Hangman-playing system that combines a lightweight character-level HMM (language priors) with a Deep Q-Network (DQN) reinforcement learner. The HMM provides per-letter probabilities given the current masked word; the DQN uses that plus the game state to choose letters to maximize long-term reward (win rate while minimizing wrong/repeat guesses).

Repository contents (files)

corpus.txt — 50k-word corpus (training data for HMM and environment word pool).

hmm_train.py — trains/saves hmm_model.joblib (HMMHybrid). Provides get_hmm_letter_probs(masked_word, guessed_set).

hangman_env.py — Gym-like Hangman environment and state-encoding utilities.

dqn_agent.py — DQN network, replay buffer, agent class (save/load, act, update).

train_agent.py — training loop: loads HMM, creates env + DQN agent, trains and saves models/dqn_best.pth / models/dqn_final.pth.

evaluate_agent.py — evaluates saved model on many games and computes the Hackman final score.

play.py — interactive console tester: give a custom word; shows top-5 predictions + normalized confidences and step-by-step guessing.

models/ — saved checkpoints (created after training).

requirements.txt — Python dependencies.

README.md — (this file).

Quick start (local or Colab)
1) Prepare environment
# recommended: Python 3.8+
pip install -r requirements.txt
# if you don't have a requirements file:
pip install numpy scipy scikit-learn joblib torch tqdm

2) Train / create HMM

If you only need the HMM (fast):

python hmm_train.py --corpus corpus.txt --out hmm_model.joblib


This produces hmm_model.joblib used by the environment and play/eval scripts.

3) Train the DQN agent

Full training (may take long; GPU recommended):

python train_agent.py


The script will create a models/ folder and save dqn_best.pth and dqn_final.pth.

It evaluates periodically and keeps the best checkpoint.

Resume training: train_agent.py checks for models/dqn_best.pth and loads it automatically. (This allows incremental training.)

4) Evaluate

Run the test suite and compute final score:

python evaluate_agent.py


Outputs: games, wins, success rate, total wrong guesses, repeated guesses, final score.

5) Interactive play (manual test)

Try a single custom word:

python play.py


It asks for a word and then shows, each turn:

Top-5 predicted letters with normalized confidence (0–1),

The chosen letter, masked word after guess, reward, and remaining lives.

How it works — short technical summary

HMMHybrid (hmm_train.py)

Builds unigram and bigram counts and stores the full corpus.

Pattern-matches corpus words to the masked word and returns a 26-dim letter-probability vector (fallback to unigram if no matches).

Saved as hmm_model.joblib.

Environment (hangman_env.py)

State vector = [masked_word_onehot (max_len * 27)] + [guessed_letters (26)] + [hmm_probs (26)].

Action space = 26 letters.

Reward shaping:

+5 correct, −2 wrong, −1 repeat, +10 win, −10 loss (tunable).

DQN (dqn_agent.py)

Neural net maps state → 26 Q-values.

Experience replay + target network + soft updates.

ε-greedy exploration with exponential decay.

Training (train_agent.py)

Plays episodes, stores transitions, updates network with minibatches.

Periodic evaluation to save best checkpoint.

Evaluation (evaluate_agent.py)

Plays fixed number of games (2000 by default) and computes final score:

Final Score = (SuccessRate * 2000) - (TotalWrongGuesses * 5) - (TotalRepeatedGuesses * 2)

Recommended hyperparameters & tips (to get best performance)

Use GPU (Colab GPU or local CUDA) for faster training.

Increase episodes in train_agent.py to 40k–100k for strong performance.

Increase network size (e.g., hidden=1024) but note: if you change architecture, you must retrain (old checkpoints won’t load).

batch_size = 256, lr = 1e-4 (or 5e-5 for smoother convergence).

Slower epsilon decay helps exploration: epsilon_decay = 15000 or more.

Use curriculum learning: warm-start with shorter words, then longer ones.

Consider Double DQN, Dueling DQN, and Prioritized Replay for better sample efficiency (future improvements).

Common troubleshooting

ModuleNotFoundError: No module named 'hmm_train'
Ensure you run scripts from the repository root where hmm_train.py is located (%cd into the project folder in Colab).

hmm_model.joblib not found
Run python hmm_train.py to generate it.

models/dqn_best.pth not found
Run python train_agent.py to train and save the model (or run a short training if you want a quick checkpoint).

Shape mismatch when loading models
If you change the DQN architecture (hidden size) after saving a checkpoint, previously-saved checkpoints will not load. Either set the agent hidden size to the value used at training time, or retrain.

play.py state-size mismatch (mat1 x mat2 shapes)
The play script pads the masked-word encoding to hmm.max_word_len * 27, so it matches the state dimension the DQN expects — make sure hmm_model.joblib is from the same corpus used for training.

Files to edit for custom behavior

hangman_env.py — change reward values, max_lives, state encoding.

dqn_agent.py — change network architecture, learning rate.

train_agent.py — change episodes, batch_size, epsilon_decay, evaluation frequency.

Evaluation & Deliverables

For the hackathon submission produce:

models/dqn_best.pth (trained weights).

hmm_model.joblib (HMM).

train_agent.py, hmm_train.py, dqn_agent.py, hangman_env.py, evaluate_agent.py, play.py (source).

Analysis_Report.pdf — include architecture choices, reward design, training curves, final score, lessons learned, and possible improvements.

Example commands summary
# install deps
pip install -r requirements.txt

# train HMM
python hmm_train.py --corpus corpus.txt --out hmm_model.joblib

# train DQN (may take a long time)
python train_agent.py

# evaluate saved model
python evaluate_agent.py

# manual play with a custom word
python play.py

Notes & attribution

The system uses only corpus.txt (no external word lists or pretrained language models).

The code uses PyTorch for DQN and simple joblib for model persistence.

