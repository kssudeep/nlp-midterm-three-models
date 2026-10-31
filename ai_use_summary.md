# AI-Use Summary

**Tools used:** Claude on claude.ai (chat), and Claude Code (Claude Opus 5.5) in VS Code.

**Conversation record:** the claude.ai chat is shared at
https://claude.ai/share/d4ce8441-8480-4148-b4ef-8579bace8c98.
The Claude Code session cannot be shared as a link, so everything it did is listed below.

## What Claude did

### In the claude.ai chat
- Wrote the first version of `midterm_lm.py`: the character tokenizer and train/validation split, the
  three models (bigram, LSTM, transformer), the shared training and evaluation code with fixed evaluation
  windows, text generation, and saving the results table, samples and loss plot.
- Wrote the templates for `README.md` and this file.

### In Claude Code
- Set up the Python environment (Python 3.13, PyTorch 2.15.0, matplotlib) on my Mac.
- Ran a 300-step test, the main 3,000-step run and the 8,000-step run.
- Added the `--out_dir` option to `midterm_lm.py` so the longer run would not overwrite the main results.
  This was the only change to the code.
- Noticed that my Mac had gone to sleep during the 8,000-step run, which made the LSTM's measured
  training time wrong. Claude restarted the run, recalculated that time from the speed after the Mac woke,
  and documented this in `runs/8000_steps/results.md` and the README.
- Added `.gitignore`, pinned the package versions in `requirements.txt` and set up the git repository.
- **Drafted all of the written text in `README.md`**: the model explanations, the Low/Medium/High quality
  ratings and notes, the six analysis answers, "What I learned", and the final reflection. It also
  drafted this summary.

## What I did
- Gave Claude the assignment and directed the work.
- Decided to run the longer 8,000-step comparison to test whether the transformer would overtake the
  LSTM after the 3,000-step run showed the LSTM ahead.
- Asked Claude to draft the written sections listed above.

## Completed by me before submitting
- Parts of the README I reviewed, edited or rewrote in my own words:
- What I can explain without help (for example, why the bigram's loss stops improving, what the causal
  mask does, why the LSTM overfit):
- How I checked that the numbers in the README match `run_log.txt` and `runs/8000_steps/run_log.txt`:

## How the comparison was kept fair
All three models use the same data, the same 90/10 split, the same batch size, context length and number
of steps, and the same random seed. Every model is evaluated on exactly the same 50 train and 50
validation batches, because the evaluation code re-seeds its own random generator each time
(`estimate_loss` in `midterm_lm.py`). Only the architecture and the learning rate differ. The full
training logs are in `run_log.txt` and `runs/8000_steps/run_log.txt`.
