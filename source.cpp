/*
	Victor Wei

	ICS4UI Final Project - Create a handwritten math expression simplifier program using image recognition (convolutional neural network)

	Sprint 1 notes:

	- Successfully completed stages 1, 3, 4, 5, 6
	- Program is able to take an image, process it, and recognize each symbol to a certain accuracy
	- Currently there is no GUI or "convolutional" aspect yet (only the fully connected neural network)
	- Attempts in implementing CUDA acceleration to make training the neural network faster

	Final submission notes:

	- All aspects of the program completed except for the camera module and GUI (out of the scope of this project)
	- Able to take a cropped handwritten expression, output the recognized symbols and simplify the recognized expression
	- Added the MOJO library to help with building the convolutional neural network
	- Added error checking for the library
	- Added a command line input (to train, read, test neural network) so it's easier to use the program
	- Removed the CUDA library (because it didn't accelerate the training)
	- Optimized and cleaned up code (that was messy in sprint 1)

	Required files / libraries to run program

	- Mnist dataset
	- Math operators dataset
	- OpenCV library
	- MOJO library

	Future notes:

	- Split the code into header files for easier management
	- Add a GUI for the user to easily interact with
	- Preferred coding style?
*/

// Include basic libraries

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// OpenCV library

#include <opencv2/opencv.hpp>

// Mojo library (neural network helper)

#define MOJO_OMP
#include "mojo/mojo.h";

// Namespace inclusion

using namespace cv;
using namespace std;
using namespace mojo;

// CLASSES AND HELPER FUNCTIONS

// Remove substring from string
void removeSubstring(std::string & mainStr, const std::string & toErase)
{
	// Search for the substring in string
	size_t pos = mainStr.find(toErase);

	if (pos != std::string::npos)
	{
		// If found then erase it from string
		mainStr.erase(pos, toErase.length());
	}
}

// Sigmoid functions

float sig(float x) // Returns a crushed number between 0 and 1
{
	return 1 / (1 + exp(-x));
}

float dSig(float x) // Returns the derivative value of the sigmoid function
{
	return x * (1 - x);
}

// RELU functions

float rectified(float x) // Returns max of 0 or x
{
	return max(static_cast<float>(0), x);
}

float dRectified(float x) // Returns 1 if x > 0 or returns 0
{
	if (x > 0)
		return 1;
	else
		return 0;
}

// Reverse RGB pixel function
int reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// MATRIX CLASS - Creates a matrix and helps perform matrix operations
class Matrix
{
public:
	int rows, cols;
	vector<vector <float> > data;

	// Constructor

	Matrix(int rows, int cols)
	{
		this->rows = rows;
		this->cols = cols;

		for (int i = 0; i < rows; i++)
		{
			data.push_back({});
			for (int j = 0; j < cols; j++)
			{
				data[i].push_back(0);
			}
		}
	}

	// Static matrix-matrix multiplication

	static Matrix multiply(Matrix a, Matrix b)
	{
		Matrix result = Matrix(a.rows, b.cols);
		if (a.cols != b.rows) // Check if matrix dimensions are correct
		{
			cout << "Cols of A don't match Rows of B";
		}
		else
		{
			// Compute the dot product of the rows of first matrix with the columns of the second matrix
			for (int i = 0; i < result.rows; i++)
			{
				for (int j = 0; j < result.cols; j++)
				{
					float sum = 0;
					for (int k = 0; k < a.cols; k++)
					{
						sum += a.data[i][k] * b.data[k][j];
					}

					result.data[i][j] = sum;
				}
			}
		}
		return result;
	}

	// Multiplication

	void multiply(Matrix n)
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->cols; j++)
			{
				this->data[i][j] *= n.data[i][j];
			}
		}
	}

	// Scalar multiplication

	void multiply(float n)
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->cols; j++)
			{
				this->data[i][j] *= n;
			}
		}
	}

	// Addition

	void add(Matrix n)
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->cols; j++)
			{
				this->data[i][j] += n.data[i][j];
			}
		}
	}

	// Scalar addition

	void add(float n)
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->cols; j++)
			{
				this->data[i][j] += n;
			}
		}
	}

	// Static subtraction

	static Matrix subtract(Matrix a, Matrix b)
	{
		Matrix result = Matrix(a.rows, a.cols);
		if (a.rows != b.rows || a.cols != b.cols)
		{
			cout << "Columns and Rows of A must match Columns and Rows of B.";
		}
		else
		{
			for (int i = 0; i < result.rows; i++)
			{
				for (int j = 0; j < result.cols; j++)
				{
					result.data[i][j] = a.data[i][j] - b.data[i][j];
				}
			}
		}
		return result;
	}

	// Subtract

	void subtract(Matrix n)
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->cols; j++)
			{
				this->data[i][j] -= n.data[i][j];
			}
		}
	}

	// Scalar subtraction

	void subtract(float n)
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->cols; j++)
			{
				this->data[i][j] -= n;
			}
		}
	}

	// Array to Matrix
	static Matrix fromArray(vector<float> arr)
	{
		Matrix m = Matrix(arr.size(), 1);
		for (int i = 0; i < arr.size(); i++)
		{
			m.data[i] = { arr[i] };
		}
		return m;
	}

	// 2D Array to Matrix
	static Matrix from2DArray(vector<vector <float> > arr)
	{
		Matrix m = Matrix(arr.size(), arr[0].size());
		for (int i = 0; i < arr.size(); i++)
		{
			for (int j = 0; j < arr[i].size(); j++)
			{
				m.data[i][j] = arr[i][j];
			}
		}
		return m;
	}

	// Matrix to Array
	vector<float> toArray()
	{
		vector<float> arr;
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->cols; j++)
			{
				arr.push_back(this->data[i][j]);
			}
		}
		return arr;
	}

	// Randomize
	void randomize()
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->cols; j++)
			{
				float random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); // Get random float from 0 to 1
				random = random * 2 - 1; // Map to range -1 to 1
				this->data[i][j] = random;
			}
		}
	}

	// Transpose
	static Matrix transpose(Matrix m)
	{
		Matrix result = Matrix(m.cols, m.rows);
		for (int i = 0; i < m.rows; i++)
		{
			for (int j = 0; j < m.cols; j++)
			{
				result.data[j][i] = m.data[i][j];
			}
		}
		return result;
	}

	// Sig Map
	void sigMap()
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->cols; j++)
			{
				this->data[i][j] = sig(this->data[i][j]);
			}
		}
	}

	// Derivative Sigmoid Map
	void dSigMap()
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->cols; j++)
			{
				this->data[i][j] = dSig(this->data[i][j]);
			}
		}
	}

	// Copy

	static Matrix copy(Matrix m)
	{
		Matrix result = Matrix(m.rows, m.cols);
		for (int i = 0; i < m.rows; i++)
		{
			for (int j = 0; j < m.cols; j++)
			{
				result.data[i][j] = m.data[i][j];
			}
		}
		return result;
	}

	// Print

	void print()
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				cout << data[i][j] << " ";
			}
			cout << "\n";
		}
	}
};

// NEURAL NETWORK CLASS - Initiates a convolutional neural network and includes functions to test and train
class NeuralNetwork
{
public:
	// Set neural network variables
	int numInput, numHidden, numOutput, filterSize, filterLength;
	float lr;
	Matrix inputWeights = Matrix(0, 0), hiddenWeights = Matrix(0, 0), hiddenBias = Matrix(0, 0), outputBias = Matrix(0, 0);
	vector<vector<vector<float> > > filters;

	// Constructor

	NeuralNetwork(int numInput, int numHidden, int numOutput)
	{
		// Fully-connected node lengths
		this->numInput = numInput;
		this->numHidden = numHidden;
		this->numOutput = numOutput;

		// Initialize and randomize the weights for the input and hidden layer
		this->inputWeights = Matrix(numHidden, numInput);
		this->hiddenWeights = Matrix(numOutput, numHidden);
		this->inputWeights.randomize();
		this->hiddenWeights.randomize();

		// Initialize and randomize the bias
		this->hiddenBias = Matrix(numHidden, 1);
		this->outputBias = Matrix(numOutput, 1);
		this->hiddenBias.randomize();
		this->outputBias.randomize();

		// Initialize and randomize the filters for the convolutional layer
		this->filterSize = 5;
		this->filterLength = 10;
		this->filters = filters;

		for (int i = 0; i < this->filterLength; i++)
		{
			vector<vector<float> > filter;
			for (int j = 0; j < this->filterSize; j++)
			{
				filter.push_back({});
				for (int k = 0; k < this->filterSize; k++)
				{
					float random = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); // Get random float from 0 to 1
					random = random * 2 - 1; // Map to range -1 to 1
					filter[j].push_back(random);
				}
			}
			this->filters.push_back(filter);
		}

		// Set learning rate
		this->lr = 0.001;
	}

	// Convolutional layer - helps detect feature / patterns in an image
	vector<Matrix> conv(vector<vector<float> > inputArray)
	{
		vector<Matrix> outputs; // Set vector of output matrices

		for (int i = 0; i < inputArray.size(); i++)
		{
			// Convert to 2D array then to matrix form
			vector<float> input = inputArray[i];
			int size = sqrt(input.size());
			vector<vector<float> > input2D;
			for (int j = 0; j < size; j++)
			{
				input2D.push_back({});
				for (int k = 0; k < size; k++)
				{
					input2D[j].push_back(input[j*size + k]);
				}
			}
			Matrix inputs = Matrix::from2DArray(input2D);

			// Set filter variables (stride, padding, output size)
			Matrix filter = Matrix(this->filterSize, this->filterSize);
			int stride = 1;
			int padding = 0;
			int outputSize = (size - filter.rows + 2 * padding) / stride + 1;

			// Apply filters to image matrix
			for (int j = 0; j < this->filters.size(); j++)
			{
				Matrix output = Matrix(outputSize, outputSize);
				Matrix filter = Matrix::from2DArray(this->filters[j]);

				// Loop through matrix
				for (int l = 0; l < output.rows; l++)
				{
					for (int k = 0; k < output.cols; k++)
					{
						float sum = 0;
						// Apply filter to matrix
						for (int m = 0; m < filter.data.size(); m++)
						{
							for (int n = 0; n < filter.data.size(); n++)
							{
								float product = filter.data[m][n] * inputs.data[k + m][l + n];
								sum += product;
							}
						}
						output.data[k][l] = sum;
					}
				}

				outputs.push_back(output);
			}
		}

		return outputs; // Return outputs
	}

	// Convolutional layer (overloading function for different input type argument)
	vector<Matrix> conv(vector<Matrix> inputArray)
	{
		vector<Matrix> outputs;
		for (int i = 0; i < inputArray.size(); i++)
		{
			// Convert to 2D array
			Matrix input = inputArray[i];
			int size = input.rows;
			vector<vector<float> > input2D = input.data;
			Matrix inputs = Matrix::from2DArray(input2D);

			Matrix filter = Matrix(this->filterSize, this->filterSize);
			int stride = 1;
			int padding = 0;

			int outputSize = (size - filter.rows + 2 * padding) / stride + 1;

			// Apply filters
			for (int j = 0; j < this->filters.size(); j++)
			{
				Matrix output = Matrix(outputSize, outputSize);
				Matrix filter = Matrix::from2DArray(this->filters[j]);

				// Loop through matrix
				for (int l = 0; l < output.rows; l++)
				{
					for (int k = 0; k < output.cols; k++)
					{
						float sum = 0;
						// Apply filter
						for (int m = 0; m < filter.data.size(); m++)
						{
							for (int n = 0; n < filter.data.size(); n++)
							{
								float product = filter.data[m][n] * inputs.data[k + m][l + n];
								sum += product;
							}
						}
						output.data[k][l] = sum;
					}
				}

				outputs.push_back(output);
			}
		}
		return outputs;
	}

	// Backpropagate through convolutional layer - Adjusts the filter weights based on the error from the previous layer
	vector<Matrix> convBack(vector<vector<float> > inputArray, Matrix inputFilter)
	{
		vector<Matrix> outputs;
		for (int i = 0; i < inputArray.size(); i++)
		{
			// Convert to 2D array then to matrix form
			vector<float> inputRaw = inputArray[i];
			int sizeRaw = sqrt(inputRaw.size());
			vector<vector<float> > input2DRaw;
			for (int j = 0; j < sizeRaw; j++)
			{
				input2DRaw.push_back({});
				for (int k = 0; k < sizeRaw; k++)
				{
					input2DRaw[j].push_back(inputRaw[j*sizeRaw + k]);
				}
			}
			// Convert to 2D array
			Matrix input = Matrix::from2DArray(input2DRaw);
			int size = input.rows;
			vector<vector<float> > input2D = input.data;
			Matrix inputs = Matrix::from2DArray(input2D);

			Matrix filter = inputFilter;
			int stride = 1;
			int padding = 0;

			int outputSize = (size - filter.rows + 2 * padding) / stride + 1;

			// Apply filters
			Matrix output = Matrix(outputSize, outputSize);

			// Loop through matrix
			for (int l = 0; l < output.rows; l++)
			{
				for (int k = 0; k < output.cols; k++)
				{
					float sum = 0;
					// Apply filter
					for (int m = 0; m < filter.data.size(); m++)
					{
						for (int n = 0; n < filter.data.size(); n++)
						{
							float product = filter.data[m][n] * inputs.data[k + m][l + n];
							sum += product;
						}
					}
					output.data[k][l] = sum;
				}
			}

			// Multiply by learning rate
			output.multiply(this->lr);

			outputs.push_back(output);
		}
		return outputs;
	}

	// RELU (Rectified Linear Unit) activation function - Helps to introduce non-linearity in the neural network
	vector<Matrix> relu(vector<Matrix> input)
	{
		vector<Matrix> outputs;
		for (int i = 0; i < input.size(); i++)
		{
			Matrix output = input[i];

			// Perform relu function to each index of the image matrix
			for (int j = 0; j < output.data.size(); j++)
			{
				for (int k = 0; k < output.data[j].size(); k++)
				{
					output.data[k][j] = rectified(output.data[k][j]);
				}
			}

			outputs.push_back(output);
		}
		return outputs;
	}

	// Max pooling function - Reduces the spatial size of the image (lowers the computational time)
	vector<Matrix> pool(vector<Matrix> input)
	{
		vector<Matrix> outputs;
		int size = 2;
		for (int i = 0; i < input.size(); i++)
		{
			Matrix old = input[i]; // Non pooled matrix
			Matrix output = Matrix(floor(old.rows / size), floor(old.rows / size)); // New size of the pooled matrix

			// Perform pooling function
			for (int j = 0; j < output.data.size(); j++)
			{
				for (int k = 0; k < output.data[j].size(); k++)
				{
					float poolValue = 0;

					// Find the maximum value of a size x size filter
					for (int l = 0; l < size; l++)
					{
						for (int m = 0; m < size; m++)
						{
							float value = old.data[k*size + l][j*size + m];
							if (value > poolValue)
							{
								poolValue = value;
							}
						}
					}

					output.data[k][j] = poolValue;
				}
			}

			outputs.push_back(output);
		}
		return outputs;
	}

	// Feed forward through neural network - Takes a 1D array of an image and propagates through the entire network to output a classification
	int feedForward(vector<float> input)
	{
		vector<vector<float >> rawInput = { input };
		vector<Matrix> conv = this->conv(rawInput); // Convolutional layer
		vector<Matrix> relu = this->relu(conv); // RELU activation layer
		vector<Matrix> pool = this->pool(relu); // Max pooling layer

		// Fully-connected layer (classifies the image to a label)

		// Flatten the vector of matrices into a 1D array
		vector<float> inputArray;
		for (int i = 0; i < pool.size(); i++)
		{
			vector<float> flat = pool[i].toArray();
			for (int j = 0; j < flat.size(); j++)
			{
				inputArray.push_back(flat[j]);
			}
		}

		// Check if the size of the input array matches the input length of the fully connected layer
		if (inputArray.size() != this->numInput)
		{
			// If the size doesn't match, change the length of the input nodes
			this->numInput = inputArray.size();
			this->inputWeights = Matrix(this->numHidden, this->numInput);
			this->inputWeights.randomize();
		}

		// Convert the vector of floats into a Matrix
		Matrix inputs = Matrix::fromArray(inputArray);

		// Multiply the weights with the inputs
		Matrix hidden = Matrix::multiply(this->inputWeights, inputs);
		// Add the bias
		hidden.add(this->hiddenBias);
		// Put through activation function
		hidden.sigMap();

		// Perform previous three steps with output
		Matrix outputs = Matrix::multiply(this->hiddenWeights, hidden);
		outputs.add(this->outputBias);
		outputs.sigMap();

		// Map the outputs to the highest label probability
		float highestIndex = 0;
		float highestProbability = 0;
		for (int i = 0; i < outputs.toArray().size(); i++)
		{
			float probability = outputs.toArray()[i];

			if (probability > highestProbability)
			{
				highestProbability = probability;
				highestIndex = i;
			}
		}

		return highestIndex;
	}

	// Backpropagates through the whole neural network - Tweaks the weights of the neural network with respect to the error from the feed forward
	void train(vector<float> input_array, vector<float> target_array)
	{
		// FEEDFORWARD (get the output values)
		vector<vector<float >> rawInput = { input_array };
		vector<Matrix> conv = this->conv(rawInput);
		vector<Matrix> relu = this->relu(conv);
		vector<Matrix> pool = this->pool(relu);

		// Flatten to a 1D array
		vector<float> inputArray;
		for (int i = 0; i < pool.size(); i++)
		{
			vector<float> flat = pool[i].toArray();
			for (int j = 0; j < flat.size(); j++)
			{
				inputArray.push_back(flat[j]);
			}
		}

		// Convert vector of floats to a Matrix
		Matrix inputs = Matrix::fromArray(inputArray);

		// Multiply the weights with the inputs
		Matrix hidden = Matrix::multiply(this->inputWeights, inputs);
		// Add the bias
		hidden.add(this->hiddenBias);
		hidden.sigMap(); // Sigmoid activation function

		// Perform previous three steps with output
		Matrix outputs = Matrix::multiply(this->hiddenWeights, hidden);
		outputs.add(this->outputBias);
		outputs.sigMap(); // Sigmoid activation function

		// BACKPROPAGATION (tweak the weights)

		// Convert target array to matrix
		Matrix targets = Matrix::fromArray(target_array);

		/*

		How to optimize the weights in a fully-connected layer

		1. Calculate error
		2. Get the gradient
		3. Reverse the activation function (derivative)
		4. Multiply by error and learning rate
		5. Transpose weights
		6. Multiply transposed weights with gradient to get the delta
		7. Add to weights to optimize
		8. Repeat to the layer on the left

		*/

		// Get error / loss of output layer
		Matrix outputErrors = Matrix::subtract(targets, outputs);

		// Get the gradient of the loss
		Matrix gradients = Matrix::copy(outputs);
		gradients.dSigMap(); // Derivative sigmoid function
		gradients.multiply(outputErrors);
		gradients.multiply(this->lr);

		// Transpose the hidden weights
		Matrix hiddenT = Matrix::transpose(hidden);

		// Get the deltas
		Matrix hiddenWeightsD = Matrix::multiply(gradients, hiddenT);

		// Change the hidden weights
		this->hiddenWeights.add(hiddenWeightsD);
		// Adjust hidden bias
		this->outputBias.add(gradients);

		// Repeat to hidden layer

		// Calculate the hidden errors
		Matrix hiddenWeightsT = Matrix::transpose(this->hiddenWeights);
		Matrix hiddenErrors = Matrix::multiply(hiddenWeightsT, outputErrors);

		Matrix hiddenGrad = Matrix::copy(hidden);
		hiddenGrad.dSigMap();
		hiddenGrad.multiply(hiddenErrors);
		hiddenGrad.multiply(this->lr);

		// Now, we calculate the deltas from input to hidden
		Matrix inputsT = Matrix::transpose(inputs);
		Matrix inputWeightsD = Matrix::multiply(hiddenGrad, inputsT);

		// Change the hidden weights
		this->inputWeights.add(inputWeightsD);
		// Adjust hidden bias
		this->hiddenBias.add(hiddenGrad);

		// Backprop Max Pooling layer

		vector<float> input = Matrix::copy(inputs).toArray();
		vector<Matrix> poolGrad;
		for (int i = 0; i < this->filterLength; i++)
		{
			// Convert to square matrix
			int size = pool[i].rows;
			vector<vector<float> > input2D;
			for (int j = 0; j < size; j++)
			{
				input2D.push_back({});
				for (int k = 0; k < size; k++)
				{
					input2D[j].push_back(input[j*size + k + i * (size * size)]);
				}
			}
			poolGrad.push_back(Matrix::from2DArray(input2D));
		}

		// Backprop RELU layer
		vector<Matrix> reluGrad = relu;
		for (int i = 0; i < reluGrad.size(); i++)
		{
			Matrix outputs = reluGrad[i];
			for (int j = 0; j < reluGrad[i].data.size(); j++)
			{
				for (int k = 0; k < reluGrad[i].data[j].size(); k++)
				{
					reluGrad[i].data[k][j] = dRectified(reluGrad[i].data[k][j]);
				}
			}
		}

		// Backprop convolutional layer

		for (int i = 0; i < reluGrad.size(); i++)
		{
			Matrix gradBack = reluGrad[i];
			vector<Matrix> convGrad = this->convBack(rawInput, gradBack); // Get the gradient of the convolutional layer

			Matrix filter = Matrix::from2DArray(this->filters[i]);
			filter.add(convGrad[0]); // Add delta to weights
			this->filters[i] = filter.data;
		}
	}

	// Print convolutional neural network model
	void print()
	{
		cout << "Input: " << numInput << " Hidden: " << numHidden << " Output: " << numOutput << "\nInput weights: \n";

		this->inputWeights.print();
		cout << "Hidden weights: \n";
		this->hiddenWeights.print();
		cout << "\n";
	}
};

// MNIST LOADER - Functions that load the MNIST dataset (handwritten digits) as well as handwritten math operator datasets

// Read MNIST dataset
void readMNIST(string filename, vector<cv::Mat> &vec) 
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		// Initialize file stream variables
		int magicNumber = 0;
		int numberOfImages = 0;
		int nRows = 0;
		int nCols = 0;

		// Set the variables through the file stream
		file.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = reverseInt(magicNumber);
		file.read((char*)&numberOfImages, sizeof(numberOfImages));
		numberOfImages = reverseInt(numberOfImages);
		file.read((char*)&nRows, sizeof(nRows));
		nRows = reverseInt(nRows);
		file.read((char*)&nCols, sizeof(nCols));
		nCols = reverseInt(nCols);

		// Convert each image from the dataset into an OpenCV image
		for (int i = 0; i < numberOfImages; ++i)
		{
			Mat image = Mat::zeros(nRows, nCols, CV_8UC1);
			// Copy each pixel into the array
			for (int r = 0; r < nRows; ++r)
			{
				for (int c = 0; c < nCols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					image.at<uchar>(r, c) = (int)temp;
				}
			}
			vec.push_back(image);
		}
	}
}

// Read MNIST labels
void readMNISTLabel(string filename, vector<double> &vec)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magicNumber = 0;
		int numberOfImages = 0;
		int nRows = 0;
		int nCols = 0;
		file.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = reverseInt(magicNumber);
		file.read((char*)&numberOfImages, sizeof(numberOfImages));
		numberOfImages = reverseInt(numberOfImages);

		// Convert the labels into a vector of doubles
		for (int i = 0; i < numberOfImages; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec[i] = (double)temp;
		}
	}
}

// Read and Preprocess Math Operator datasets
void readDataset(string dataset_path, vector<Mat> &vec, vector<double> &vecLabel, double label)
{
	// Set preprocessing variables
	int dilationSize = 2;
	int borderSize = 5;
	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * dilationSize + 1, 2 * dilationSize + 1),
		Point(dilationSize, dilationSize));

	cv::String path(dataset_path); // Set path of the dataset
	vector<cv::String> rawData;
	vector<cv::Mat> data;
	cv::glob(path, rawData, true);
	for (size_t k = 0; k < rawData.size(); ++k)
	{
		// Convert into 28 x 28, dilated, and binarized image
		cv::Mat im = cv::imread(rawData[k]);

		cvtColor(im, im, COLOR_BGR2GRAY);
		bitwise_not(im, im);
		copyMakeBorder(im, im, borderSize, borderSize, borderSize, borderSize, BORDER_CONSTANT); // Add border to image
		dilate(im, im, element);
		copyMakeBorder(im, im, borderSize, borderSize, borderSize, borderSize, BORDER_CONSTANT); // Add border to image
		resize(im, im, Size(28, 28));

		//if (im.empty()) continue; // Proceed if successful
		data.push_back(im);
	}

	int length = std::min(1000, static_cast<int>(data.size()));

	// Add on math symbols
	for (int i = 0; i < length; i++)
	{
		vec.push_back(data[i]);
		vecLabel.push_back(label);
	}
}

// IMAGE PROCESSING - Preprocesses the handwritten image so it can be easily segmented

// Variables for processing
RNG rng(12345);
int thresh = 100;
int thresholdValue = 150;
int maxBinaryValue = 255;

Mat processImage(Mat src)
{
	//imshow("original", src); // Show original image

	// Convert image to greyscale (black and white image)
	Mat srcGray;
	cvtColor(src, srcGray, COLOR_BGR2GRAY);
	blur(srcGray, srcGray, Size(3, 3));
	//imshow("gray", srcGray); // Show greyscale image

	// Threshold image (binarization)
	threshold(srcGray, srcGray, thresholdValue, maxBinaryValue, THRESH_BINARY);

	// Dilate the image (make the symbols thicker)
	int dilationSize = 5;
	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * dilationSize + 1, 2 * dilationSize + 1),
		Point(dilationSize, dilationSize));
	erode(srcGray, srcGray, element);

	return srcGray;
}

// IMAGE SEGMENTATION - Segments the image into individual symbols to later classify

// Sorts contours from left to right
struct contourSorter
{
	bool operator ()(const vector<Point>& a, const vector<Point> & b)
	{
		Rect ra(boundingRect(a));
		Rect rb(boundingRect(b));
		return (ra.x < rb.x);
	}
};

// Segment image into individual symbols
vector<cv::Mat> segmentImage(int, void*, Mat srcGray)
{
	// Initialize vector of segmented images and rectangles
	vector<cv::Mat> segmented;
	vector<Rect> segmentedRect;

	// Find contours (edge detection)
	Mat cannyOutput;
	Canny(srcGray, cannyOutput, thresh, thresh * 2); // Detect edges
	vector<vector<Point> > contours;
	vector< Vec4i > hierarchy;
	findContours(cannyOutput, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	// Sort contours from left to right
	sort(contours.begin(), contours.end(), contourSorter());

	// Set bounding rectangle variables
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	if (contours.size() > 0)
	{
		// Find the bounding rectangles (around the contours)
		for (size_t i = 0; i < contours.size(); i++)
		{
			approxPolyDP(contours[i], contours_poly[i], 3, true);
			boundRect[i] = boundingRect(contours_poly[i]);
		}

		// Filter out extraneous bounding rectangles (duplicates, small particles, etc.)
		Mat drawing = Mat::zeros(cannyOutput.size(), CV_8UC3);
		for (size_t i = 0; i < contours.size(); i++)
		{
			// Ignore if this bounding box is in each other bounding boxes
			bool inside = false;
			for (size_t j = 0; j < contours.size(); j++)
			{
				if (i != j)
				{
					Rect a = boundRect[i];
					Rect b = boundRect[j];
					if (a.x > b.x && a.x + a.width < b.x + b.width && a.y > b.y && a.y + a.height < b.y + b.height)
					{
						inside = true;
					}
				}
			}

			// Ignore boxes formed by noise and interferes with previous contours
			float area = boundRect[i].width*boundRect[i].height;
			if (area > 500 && !inside)
			{
				// Check for duplicate contours
				bool duplicate = false;
				for (size_t j = 0; j < segmentedRect.size(); j++)
				{
					double distance = sqrt(pow(boundRect[i].x - segmentedRect[j].x, 2) + pow(boundRect[i].y - segmentedRect[j].y, 2)); // Distance between the two contours
					if (distance < 10)
					{
						duplicate = true;
					}
				}

				// Process image if it is not a duplicate
				if (!duplicate)
				{
					// Crop image based on bounding box
					Mat cropped;
					Point centerRect = (boundRect[i].br() + boundRect[i].tl())*0.5; // Find center of bounding box
					getRectSubPix(srcGray, boundRect[i].size(), centerRect, cropped); // Crop image
					//imshow("Cropped", cropped); // Show cropped image

					// Add padding to make the image a square (so it can be put into the neural network)
					Mat padded;
					int padding = static_cast<int>((cropped.rows - cropped.cols) / 2);
					if (padding < 0)
						copyMakeBorder(cropped, padded, abs(padding), abs(padding), 0, 0, BORDER_CONSTANT, Scalar(255));
					else
						copyMakeBorder(cropped, padded, 0, 0, abs(padding), abs(padding), BORDER_CONSTANT, Scalar(255));
					copyMakeBorder(padded, padded, 50, 50, 50, 50, BORDER_CONSTANT, Scalar(255)); // Add border to iamge
					//imshow("Padded", padded); // Show padded image

					// Dilate the image (make the symbol thicker for easier recognition)
					Mat dilated;
					int dilationSize = 1;
					Mat element = getStructuringElement(MORPH_RECT,
						Size(2 * dilationSize + 1, 2 * dilationSize + 1),
						Point(dilationSize, dilationSize));
					erode(padded, dilated, element);
					//imshow("Dilated", dilated); // Show dilated image

					// Resize image to 28 x 28 (to fit the required specifications of the neural network)
					Mat dst;
					resize(dilated, dst, Size(28, 28), 0, 0, INTER_AREA);
					segmented.push_back(dst);
					segmentedRect.push_back(boundRect[i]);
					//imshow("Segmented Image " + to_string(i), dst); // Show segmented image

					// Show all segmented images
					Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
					drawContours(drawing, contours_poly, (int)i, color);
					rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
					imshow("Segmentation", drawing); // Show the original image with bounding boxes
				}
			}
		}
	}

	return segmented;
}

// EXPRESSION SIMPLIFIER - Takes a vector of numbers and operators and simplifies it into one value

// Checks if string is a float
bool isFloat(string myString) {
	std::istringstream iss(myString);
	float f;
	iss >> noskipws >> f; // noskipws considers leading whitespace invalid

	// Check the entire string was converting into a float and if the stringstream failed
	return iss.eof() && !iss.fail();
}

// Expression simplifier
string solveExpression(vector<string> expression)
{
	while (expression.size() > 1)
	{
		// Find BEDMAS Prioritization (highest level has highest priority)
		int level = -1;
		for (int i = 0; i < expression.size(); i++)
		{
			if (expression[i] == "(")
			{
				if (level <= 3) level = 3;
			}
			else if (expression[i] == "^")
			{
				if (level <= 2) level = 2;
			}
			else if (expression[i] == "x" || expression[i] == "*" || expression[i] == "/")
			{
				if (level <= 1) level = 1;
			}
			else if (expression[i] == "+" || expression[i] == "-")
			{
				if (level <= 0) level = 0;
			}
		}

		// No operators found -> Return error
		if (level == -1)
			return "Error: Invalid expression detected\n";

		// Perform operations based on the higest level / priority
		for (int i = 0; i < expression.size(); i++)
		{
			stringstream ss;
			if (expression[i] == "(" && level == 3) // Simplify brackets
			{
				int index = i;
				// Find the bracket pair
				int found = 0;
				while (found >= 0)
				{
					index++;
					if (index >= expression.size()) // Out of bounds exception
					{
						return "Error: No paired bracket";
					}
					else if (expression[index] == "(") // Find the leftmost and outermost bracket pair first 
					{
						found++;
					}
					else if (expression[index] == ")")
					{
						found--;
					}
				}

				// Get the expression within the bracket pair
				vector<string> subExpression;
				for (int j = 0; j < index - i - 1; j++)
				{
					subExpression.push_back(expression[i + 1 + j]);
				}

				expression[i] = solveExpression(subExpression); // Simplify the sub expression

				// Remove the bracket pair
				for (int j = 0; j < index - i; j++)
				{
					expression.erase(expression.begin() + (i + 1));
				}

				break;
			}
			else if (level < 3) // Non-bracket operations
			{
				// Make sure the left and right of the operator are numbers, not operators or out of bounds
				bool isOperator = false;
				if (expression[i] == "^" && level == 2) // Simplify exponents
				{
					ss << pow(strtof((expression[i - 1]).c_str(), 0), strtof((expression[i + 1]).c_str(), 0));
					isOperator = true;
				}
				else if ((expression[i] == "x" || expression[i] == "*") && level == 1) // Simplify multiplication
				{
					ss << strtof((expression[i - 1]).c_str(), 0)*strtof((expression[i + 1]).c_str(), 0);
					isOperator = true;
				}
				else if (expression[i] == "/" && level == 1) // Simplify division
				{
					ss << strtof((expression[i - 1]).c_str(), 0) / strtof((expression[i + 1]).c_str(), 0);
					isOperator = true;
				}
				else if (expression[i] == "+" && level == 0) // Simplify addition
				{
					ss << strtof((expression[i - 1]).c_str(), 0) + strtof((expression[i + 1]).c_str(), 0);
					isOperator = true;
				}
				else if (expression[i] == "-" && level == 0) // Simplify subtraction
				{
					ss << strtof((expression[i - 1]).c_str(), 0) - strtof((expression[i + 1]).c_str(), 0);
					isOperator = true;
				}

				// If operator exists, set operator to the calculated value and remove the left and right numbers
				if (isOperator)
				{
					string s(ss.str());
					expression[i] = s;
					expression.erase(expression.begin() + i - 1);
					expression.erase(expression.begin() + i);
					break;
				}
			}
		}

		// Print each simplification of the expression
		for (int i = 0; i < expression.size(); i++)
			cout << expression[i];
		cout << "\n";
	}

	return expression[0];
}

// TRAIN NEURAL NETWORK - Loads the datasets and trains the chosen neural network to classify handwritten digits and math operators

void trainNetwork(string commandInput, network cnn, NeuralNetwork nn)
{
	// Setup MNIST images
	string filenameImage = "mnist/t10k-images-idx3-ubyte";
	int numberOfImages = 10000; // 10000 images with labels to train

	// Read MNIST image into OpenCV Mat vector
	vector<cv::Mat> vec;
	cout << "Loading MNIST dataset...\n";
	readMNIST(filenameImage, vec);

	// Setup MNIST labels

	string filenameLabel = "mnist/t10k-labels-idx1-ubyte";

	// Read MNIST label into double vector
	vector<double> vecLabel(numberOfImages);
	readMNISTLabel(filenameLabel, vecLabel);

	// Setup Math Operator datasets

	cout << "Loading math operators...\n";
	readDataset("symbols/+/*.jpg", vec, vecLabel, static_cast<double>(10));
	readDataset("symbols/-/*.jpg", vec, vecLabel, static_cast<double>(11));
	readDataset("symbols/times/*.jpg", vec, vecLabel, static_cast<double>(12));
	readDataset("symbols/slash/*.jpg", vec, vecLabel, static_cast<double>(13));
	readDataset("symbols/(/*.jpg", vec, vecLabel, static_cast<double>(14));
	readDataset("symbols/)/*.jpg", vec, vecLabel, static_cast<double>(15));
	readDataset("symbols/Delta/*.jpg", vec, vecLabel, static_cast<double>(16));

	// Flatten Matrix into vector
	vector<vector<float> > images;
	for (int i = 0; i < vec.size(); i++)
	{
		vector<float> image;
		cv::Mat flat = vec[i].reshape(1, vec[i].total()*vec[i].channels());
		image = vec[i].isContinuous() ? flat : flat.clone();
		images.push_back(image);
	}

	// Create labels
	vector<vector<float> > labels;
	int symbols = 17;
	for (int i = 0; i < symbols; i++)
	{
		vector<float> label;
		labels.push_back(label);
		for (int j = 0; j < symbols; j++)
		{
			labels[i].push_back(0);
		}
		labels[i][i] = 1;
	}

	cout << "Training dataset length: " << vec.size() << "\n";

	// Shuffle training data and labels (in the same order)
	vector<vector<float> > shuffledImages;
	vector<double> shuffledLabels;

	// Shuffle indices first
	std::vector<int> indexes;
	indexes.reserve(images.size());
	for (int i = 0; i < images.size(); ++i)
		indexes.push_back(i);
	std::random_shuffle(indexes.begin(), indexes.end());

	// Shuffle images based on the shuffled indices
	for (std::vector<int>::iterator it1 = indexes.begin(); it1 != indexes.end(); ++it1)
		shuffledImages.push_back(images[*it1]);

	// Shuffle labels based on the shuffled indices
	for (std::vector<int>::iterator it1 = indexes.begin(); it1 != indexes.end(); ++it1)
		shuffledLabels.push_back(vecLabel[*it1]);

	images = shuffledImages;
	vecLabel = shuffledLabels;

	cout << "Done shuffling data\n";

	// Choose which neural network to train
	if (commandInput.find("mojo") != string::npos) // Use MOJO neural network
	{
		removeSubstring(commandInput, "mojo");
		commandInput.pop_back();

		// Train MOJO neural network

		cnn.set_smart_training(true);
		cnn.enable_external_threads(128);
		cnn.enable_internal_threads(128);
		cnn.set_mini_batch_size(24);

		if (commandInput.find("basic") != string::npos) // Create basic convolutional neural network architecture
		{
			removeSubstring(commandInput, "basic");

			cnn.push_back("I1", "input 28 28 1");				 // MNIST is 28x28x1
			cnn.push_back("C1", "convolution 5 20 1 elu");		 // 5x5 kernel, 20 maps, stride 1.  out size is 28-5+1=24
			cnn.push_back("P1", "semi_stochastic_pool 4 4");	 // pool 4x4 blocks, stride 4. out size is 6
			cnn.push_back("C2", "convolution 5 200 1 elu");		 // 5x5 kernel, 200 maps.  out size is 6-5+1=2
			cnn.push_back("P2", "semi_stochastic_pool 2 2");	 // pool 2x2 blocks. out size is 2/2=1
			cnn.push_back("FC1", "fully_connected 100 identity");// fully connected 100 nodes
			cnn.push_back("FC2", "softmax 17");
		}
		else if (commandInput.find("deep") != string::npos) // Create deep CNet Model
		{
			removeSubstring(commandInput, "deep");

			cnn.push_back("I1", "input 28 28 1 identity");
			cnn.push_back("C1", "convolution 3 40 1 elu");
			cnn.push_back("P1", "max_pool 2 2");
			cnn.push_back("N1", "concatenate 15 15");
			cnn.push_back("D1", "deepcnet 80 elu");
			cnn.push_back("R1", "dropout 0.500000");
			cnn.push_back("D2", "deepcnet 160 elu");
			cnn.push_back("R2", "dropout 0.500000");
			cnn.push_back("D3", "deepcnet 320 elu");
			cnn.push_back("R3", "dropout 0.500000");
			cnn.push_back("F1", "fully_connected 17 softmax");
		}
		commandInput.pop_back();

		cnn.connect_all(); // Connect all layers automatically

		cout << cnn.get_configuration() << "\n";

		// Continuously train neural network until necessary to stop
		while (1)
		{
			// Start training an epoch
			cnn.start_epoch("cross_entropy");

			// Parallel computing
#pragma omp parallel
#pragma omp for schedule(dynamic)
			for (int k = 0; k < images.size(); k++) cnn.train_class(images[k].data(), vecLabel[k]);

			cnn.end_epoch();

			cout << "Estimated accuracy: " << cnn.estimated_accuracy << "%\n"; // Output estimated accuracy

			cnn.write("./models/" + commandInput); // Save current model to destination

			if (cnn.elvis_left_the_building()) break; // Exit training when needed (> 1000 epochs, or accuracy decreased 3 times in a row)
		};
	}
	else if (commandInput.find("basic") != string::npos) // Use self-made neural network
	{
		removeSubstring(commandInput, "basic");
		commandInput.pop_back();

		// Train basic / self-made neural network
		int epochs;
		std::istringstream iss(commandInput);
		iss >> epochs;

		cout << "\nTraining neural network for " << epochs << " epochs \n\n";

		if (epochs > 0)
		{
			for (int j = 0; j < epochs; j++) // Train for the set amount of epochs
			{
				float correct = 0;
				int iterations = images.size();

				for (int i = 0; i < iterations; i++)
				{
					// Print progress every 1000 iterations
					if (i % 1000 == 0)
						cout << "Iteration: " << i << "\n";

					// Check for accuracy and train based off of a random index
					int randomIndex = rand() % images.size();
					if (nn.feedForward(images[randomIndex]) == static_cast<int>(vecLabel[randomIndex]))
						correct++;

					nn.train(images[randomIndex], labels[static_cast<int>(vecLabel[randomIndex])]);
				}
				cout << "Epoch #" << j + 1 << " | Percentage correct: " << (correct / images.size() * 100) << " %\n";
			}
		}
	}
}

// TEST IMAGE - Inputs a handwritten math expression image, processes, segments, classifies, and simplifies the image

void testImage(network &cnn, NeuralNetwork &nn, string commandInput)
{
	// Process command line input
	string testType;
	if (commandInput.find("basic") != string::npos)
	{
		testType = "basic";
		removeSubstring(commandInput, "basic");
	}
	else if (commandInput.find("mojo") != string::npos)
	{
		testType = "mojo";
		removeSubstring(commandInput, "mojo");
	}
	commandInput.pop_back();

	// Read image from file path
	cout << "Loading equation image: " << commandInput << "\n";
	Mat src = imread("C:/Users/undef/Desktop/" + commandInput);

	// Image processing
	Mat processed = processImage(src);

	// Image segmentation
	vector<cv::Mat> segmented = segmentImage(0, 0, processed);

	// Classify segmented images to a label
	cout << "\nClassifying expression to labels... \n\n";
	vector<int> classifications;
	for (size_t i = 0; i < segmented.size(); i++)
	{
		Mat symbol = segmented[i]; // Get test image from the segmentated equation
		bitwise_not(symbol, symbol);
		vector<float> imageSegment;
		//imshow("Test Image" + to_string(i), symbol); // Show each segmented image
		imageSegment.assign(symbol.data, symbol.data + symbol.total());

		// Classify each image using neural network
		int index;
		if (testType == "basic") // Use self-trained CNN
			index = nn.feedForward(imageSegment);
		else if (testType == "mojo") // Use MOJO CNN
			index = cnn.predict_class(imageSegment.data());
		cout << "Image " << i << ": " << index << "\n";

		// Add classified index to vector
		classifications.push_back(index);
	}

	// Convert classified labels into an expression
	vector<string> expression;
	string expressionString = "";
	string number = "";
	string prevSymbol = "";
	for (int i = 0; i < classifications.size(); i++)
	{
		if (classifications[i] >= 10 && number.length() > 0) // Add digits from before
		{
			expression.push_back(number);
			number = "";
			if (classifications[i] == 14) // Put multiplication symbol between number and (
				expression.push_back("x");
		}

		if (prevSymbol == ")" && classifications[i] == 14) // Put multiplication symbol in between ) and (
			expression.push_back("x");

		string symbol = "";
		if (classifications[i] < 10) // Digit
		{
			number += to_string(classifications[i]);
			expressionString += to_string(classifications[i]);
		}
		else if (classifications[i] == 10) // Add
			symbol = "+";
		else if (classifications[i] == 11) // Minus
			symbol = "-";
		else if (classifications[i] == 12) // Times
			symbol = "x";
		else if (classifications[i] == 13) // Divide
			symbol = "/";
		else if (classifications[i] == 14) // Opening bracket
			symbol = "(";
		else if (classifications[i] == 15) // Closing bracket
			symbol = ")";
		else if (classifications[i] == 16) // Exponent
			symbol = "^";

		if (symbol.length() > 0)
		{
			expression.push_back(symbol);
			expressionString += symbol;
			prevSymbol = symbol;
		}
	}

	// Add leftover number
	if (number.length() > 0)
		expression.push_back(number);

	// Simplify the expression
	cout << "Simplifying expression... \n" << expressionString << "\n";
	string result = solveExpression(expression);

	// Output result if there are no errors
	if (isFloat(result))
		cout << "Answer: " << result << "\n\n";
	else
		cout << result << "\n";
}

int main(int argc, char* argv[])
{
	// Initalize random seed (for random number generator)
	srand(static_cast <unsigned> (time(0)));

	// Initiate MOJO neural network
	mojo::network cnn("adam");

	// Initiate basic neural network
	int symbols = 17;
	NeuralNetwork nn = NeuralNetwork(1440, 200, symbols);

	/*
		COMMAND LINE INTERFACE - User interface in the command prompt that allows the user to easily test, train, and read the neural networks

		test <neural network type> <image filepath> --> Detects an image and simplifies its handwritten expression
		read <model filepath>						--> Loads neural network model from file
		clear										--> Clear loaded model
		train mojo <model type> <destination path>	--> Train mojo neural network with given model type (basic or deep)
		train basic <epoch iterations>				--> Train basic neural network for a given number of iterations
		exit 										--> Exits the program
	*/

	string commandInput;
	while (commandInput != "exit") // Keeps running command line interface until user types "exit"
	{
		// Get command from user
		cout << "Enter a command: \n";
		getline(cin, commandInput);

		if (commandInput.find("test") != string::npos)
		{
			// Process command
			removeSubstring(commandInput, "test");
			remove_if(commandInput.begin(), commandInput.end(), isspace);
			commandInput.pop_back();

			// Test Image (Attempt to simplify expression from image)
			string equationPath = "C:/Users/undef/Desktop/" + commandInput;
			testImage(cnn, nn, commandInput);
		}
		else if (commandInput.find("read") != string::npos)
		{
			// Process command
			removeSubstring(commandInput, "read");
			remove_if(commandInput.begin(), commandInput.end(), isspace);
			commandInput.pop_back();

			// Read neural network model
			cout << "Reading model: " << commandInput << "\n";
			cnn.read("./models/" + commandInput); // Pre-trained model
		}
		else if (commandInput.find("clear") != string::npos)
		{
			// Clear neural network model
			cnn.clear();
			cout << "Cleared current model\n";
		}
		else if (commandInput.find("train") != string::npos)
		{
			// Process command
			removeSubstring(commandInput, "train");
			remove_if(commandInput.begin(), commandInput.end(), isspace);
			commandInput.pop_back();

			// Train neural network
			trainNetwork(commandInput, cnn, nn);
		}
		else if (commandInput != "exit")
		{
			cout << "Invalid command\n";
		}
	}

	waitKey(); // Exit on key

	return 0;
}