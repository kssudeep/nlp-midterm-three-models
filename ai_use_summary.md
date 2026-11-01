# AI-Use Summary

## Tools Used
Claude (claude.ai and Claude Code in VS Code).

**Conversation Record:** The claude.ai conversation is available [here](https://claude.ai/share/d4ce8441-8480-4148-b4ef-8579bace8c98). The Claude Code session cannot be shared as a link, so its contributions are documented below.

## How I Used AI

I used Claude as a supporting tool throughout the project, mainly for coding assistance, debugging, understanding model architectures, and improving documentation.

### Project Development
- Used Claude's assistance to develop the initial implementation of `midterm_lm.py`, including the character tokenizer, data splitting, Bigram, LSTM, and Transformer models.
- Reviewed the implementation and directed the development according to the assignment requirements.
- Used Claude Code to configure the Python environment and run the experiments.
- Requested additional experiments to compare model performance at different training durations.
- Decided to conduct an extended 8,000-step experiment after observing the results from the initial 3,000-step comparison.
- Reviewed the experiment outputs, training logs, generated text, and comparison results.

### Documentation and Improvements
- Used AI-generated README and documentation drafts as a starting point.
- Reviewed the explanations, model comparisons, and final reflection.
- Used Claude Code to organize the repository, configure dependencies, and add Git-related files.
- Reviewed the final project structure and supporting result files.

## My Contributions

My primary contributions included:
- Understanding the assignment requirements and defining the project direction.
- Making decisions about the models, experiments, and additional training runs.
- Evaluating the results and considering how training duration affected model performance.
- Reviewing the generated samples, loss values, and training logs.
- Checking the consistency of the reported results across experiment files.
- Reviewing the technical explanations and ensuring I could explain the main concepts behind the three architectures.

## Understanding and Learning

Through this project, I focused on understanding:
- How a Bigram model predicts the next character using only the current character.
- How an LSTM uses information from previous characters to learn sequential patterns.
- How self-attention and causal masking work in a Transformer.
- Why Transformers can capture relationships between tokens more effectively.
- How training duration, model complexity, and overfitting affect language modeling performance.

## Fair Model Comparison

All three models were trained using the same dataset, 90/10 training-validation split, batch size, context length, and number of training steps within each comparison. The same random seed and fixed evaluation windows were used to maintain consistency. The architectures and learning rates differed.

The training logs and experiment results are included in the repository for reference.

## Final Note

Claude assisted with the initial code, experiment execution, and documentation drafts. I directed the project, made the experimental decisions, reviewed the results, and used the project to develop my understanding of the three language-modeling architectures. I take responsibility for reviewing and understanding the final submission.