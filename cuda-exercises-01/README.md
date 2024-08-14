[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/S2D0Hvug)
# CUDA Labs

This repository contains a set of three labs focused on CUDA parallel programming, split into separate folders, each containing a specific lab exercise.

## Lab Exercise Structure

Each lab is located in a separate folder and contains the necessary files and instructions to complete it. The instructions for each lab exercise are located in a README file in the respective exercise folder.

The labs are designed to be completed in order, as each lab builds upon the previous one. Therefore, it is recommended that you start with the first lab and work your way through them in sequence.

## First: clone on local machine

Below are instructions to run this repository locally. We recommend using the Google Colab method above, and we only offer support to users using the Google Colab execution method.

To get started, clone this repository to your local machine using the following command:

```sh
git clone --recurse-submodules <repository URL>
```

The command `--recurse-submodules` is used to clone the test folder, which is from a separate repository, because of its size.

If you have already cloned the repository without the --recurse-submodules option, you can still clone the submodules by running the following command in the repository's root directory:

```sh
git submodule update --init --recursive
```

Once the repository is cloned, navigate to the first lab folder and read the instructions in the README file. Follow the instructions to complete the lab, and then move on to the next lab in the sequence.

## Running on Google Colab

1. We will use Google Drive for storage. Go to [Drive](https://drive.google.com/drive) and upload this repository folder. Tip: put in the root of your Drive, in a folder called ```cuda-exercises-01```.
2. Access [Google Colab](https://colab.research.google.com/)
3. Go to File -> Upload notebook -> Browse. Select the file ```colab-runner.ipynb``` from this repository.
4. Run the first cell and authorize Google Drive access.
5. Select the GPU runtime (NVIDIA T4): Runtime -> Change runtime type -> T4 GPU -> Save
6. Runtime -> Run all -> Run anyway

This will compile ant test your code! Check the output in the cell.

**Important**: remember to copy your solutions to your repository and commit them!

## Prerequisites

To complete these labs, you will need a basic understanding of C or C++ programming language and the fundamentals of parallel programming concepts. You will also need a development environment for the language used in the labs and CUDA libraries installed, which is already installed in the Google Colab environment.

## Contributing

If you find any issues with the labs or have suggestions for improvements, feel free to create an issue or pull request on this repository.
