- Implemented basic back propagation while learning it from karpathy's [video](https://youtu.be/VMj-3S1tku0?si=lUwaGDpuZhPbZ29A) and [repository](https://github.com/karpathy/micrograd)
1. the learning notebook contains the code used while implementing the logic in parallel with the video.
2. the engine+nn file separates the main logic from the test cases.
3. the demo file implements a simple binary classification task using micro-grad. (in karpathy's repo too)
- graphviz was not working on my local setup, so the learning notebook does not contain the DAG visualizations. It worked on google colab though, so might add that code in future.
- The values of the training data should be objects of class Value before being passed to the model.
