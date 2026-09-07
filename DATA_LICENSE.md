# Data attribution

The planned production training source is BANKING77 from PolyAI's task-specific datasets repository. It is licensed under Creative Commons Attribution 4.0 International. The publisher reports 10,003 training examples, 3,080 test examples, and 77 intents.

- Source: https://github.com/PolyAI-LDN/task-specific-datasets
- Dataset card: https://huggingface.co/datasets/PolyAI/banking77
- Citation: Casanueva et al., “Efficient Intent Detection with Dual Sentence Encoders,” 2020.

The current `train-demo` command uses a tiny, clearly labeled in-repository demonstration corpus so the local application can boot before BANKING77 is fetched. It is not a substitute for the licensed dataset and its metrics must not be reported as benchmark results.
