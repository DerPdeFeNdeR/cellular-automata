# chatGPT Game of Life Using GPU Parrallelization

## Description

This is a work in progress application built using chatGPT (and me). It is a program written in Go that uses the termbox library to display the results of a GPU-parallelized calculation in real-time. The calculation is performed using a CUDA module compiled from PTX (parallel thread execution) assembly code.

## Prerequisites

- Go
- termbox-go library (`go get github.com/nsf/termbox-go`)
- cuda library (`go get github.com/unixpickle/cuda`)
- A CUDA-compatible GPU and drivers

## Installation

1. Clone the repository to your local machine
1. Install the necessary dependencies using the commands above
1. Compile the PTX assembly code into a CUDA module and place the resulting .ptx file in the specified location in `game-of-life.go`
1. Build the Go program with `go build game-of-life.go`

## Usage

Run the compiled Go program to see the results of the GPU-parallelized calculation displayed in real-time on the terminal.

## Contributing

If you would like to contribute to the project, please follow these guidelines:

- Fork the repository
- Create a new branch for your changes
- Make your changes and commit them to your branch
- Push your branch to your fork and submit a pull request for review

## License

This project is licensed under the Apache License 2.0.