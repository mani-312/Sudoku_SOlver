# Sudoku_Solver
We often get struck at solving sudoku and try to know the answer. Though there are several online platforms to solve the sudoku and even you can write a program to solve. But the irritating part is, you need to provide input as digits at corresponding positions.
Here we use computer vision to automatically process the sudoku image and get the each digit perspective of sudoku and find the solution.
Hey, you've got the sudoku solution just by uploading pic of image.

## High-level flow of solving sudoku
- Get the input as sudoku image
- Find the sudoku perspective in the image
- Get through the each digit and recognize the digit
- Solve the sudoku

## Train Model to recognize the digit
### digit_recognizer
-          |
-          |---> mnist.npz (Dataset)
-          |---> train_model.py (Python script to train CNN model on mnist data)
-          |---> model.h5 (Saved model)
          
## Sudoku_program_flow
![alt text](https://github.com/mani-312/Sudoku_SOlver/blob/main/Sudoku_flow/Sudoku_program_flow.png?raw=true)

## Example of a sudoku_flow image
![alt text](https://github.com/mani-312/Sudoku_SOlver/blob/main/Sudoku_flow/Example.png?raw=true)
- Now grab the each digit in sudoku grid and recognize the digit
## Sudoku_digit flow
![alt text](https://github.com/mani-312/Sudoku_SOlver/blob/main/digit_flow/digit_program_flow.png?raw=true)

## Example digit_flow
![alt text](https://github.com/mani-312/Sudoku_SOlver/blob/main/digit_flow/example1.png?raw=true)

- Finally we've got all the digits in sudoku image
- I've used backtracking to solve the sudoku

## Demo
![alt text](https://github.com/mani-312/Sudoku_SOlver/blob/main/Sudoku_solved.png?raw=true)
