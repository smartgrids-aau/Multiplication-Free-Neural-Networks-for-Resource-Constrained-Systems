#	Multiplication-Free Neural Networks for Resource-Constrained Systems
Neural Networks  •  6502 Assembly  •  Embedded AI  •  Hardware Efficiency

Conventional neural networks rely heavily on floating-point multiplication for both inference and training. On resource-constrained platforms — such as 8-bit microprocessors — multiplication is either unavailable in hardware or prohibitively expensive in clock cycles.  This project designs and evaluates a multiplication-free neural network architecture suitable for inference on severely limited hardware. The core idea is to replace neuron weight multiplication with a two-step logarithm/anti-logarithm scheme using lookup tables (LUTs): convert a neuron's output to its logarithm (8-bit LUT), add the pre-stored log-weight, then convert the result back via an exponential LUT. This reduces multiplication to two LUT lookups and an integer addition — operations that are fast even on a 6502 CPU.  The architecture will be implemented and benchmarked on a standard classification task, then cross-compiled for a 6502-based platform using the oscar64 C compiler, enabling a direct comparison of footprint, accuracy, and execution speed between a standard ANN and the multiplication-free variant.

## Research Questions
*	How much classification accuracy is lost when replacing floating-point multiplication with an 8-bit log/exp LUT scheme?
*	What is the execution time of a full inference pass on a 6502 at 1 MHz, and how does it compare to a PC baseline?
*	What is the minimum network size that achieves acceptable task performance within the memory constraints of an 8-bit platform?

## Tasks
*	Design the LUT-based multiplication-free neuron: 8-bit log-LUT for neuron output, 8-bit weight storage as pre-computed log-weights, 8-bit exp-LUT for result reconstruction.
*	Implement the architecture in C (targeting oscar64) and Python/NumPy (for baseline comparison).
*	Select a standard benchmark task suitable for an 8-bit inference context (e.g., small-scale classification or pattern recognition).
*	Train the multiplication-free network and a standard ANN on the benchmark task; compare classification accuracy.
*	Cross-compile the C implementation with oscar64 for the C64/6502 architecture; measure execution time and code/data footprint.
*	Compare results across three dimensions: accuracy on the task, execution time (PC vs. 6502), and memory footprint.

## Reading List
*	Miyashita, D., Lee, E.H., Murmann, B. "Convolutional Neural Networks using Logarithmic Data Representation." arXiv:1603.01025, 2016.
*	Courbariaux, M., Bengio, Y., David, J.P. "Training Deep Neural Networks with Low Precision Multiplications." ICLR Workshop, 2015.
