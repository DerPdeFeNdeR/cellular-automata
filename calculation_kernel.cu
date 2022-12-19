__global__ void calculationKernel(int* input) {
  // Perform the calculation on the input value
  int result = input[0] * 2;

  // Store the result in the input memory location
  input[0] = result;
}