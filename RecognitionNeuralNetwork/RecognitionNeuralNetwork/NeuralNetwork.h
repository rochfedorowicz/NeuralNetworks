#pragma once
#include "Matrix2D.h"

//Class used to represent Neural Network
class NeuralNetwork {

	//Sizes of layers in network
	std::vector<int> layerSizes;

	//Weights' shapes between layers in network
	std::vector<std::pair<int, int>> weightShapes;

	//Weights between layers in network
	std::vector<Matrix2D<double>> weights;

	//Biases added to layers in network
	std::vector<std::vector<double>> biases;

	//Function to calculate sygmoid's function value from given argument
	double sygmoidFn(double val);

	//Function to calculate value of sygmoid's function derivative from given argument
	double sygmoidPrimeFn(double val);

	//Activates vector using sygmoid's function
	void activation(std::vector<double>& _inputVector);

	//Activates vector using derivative of sygmoid's function
	void reversActivation(std::vector<double>& _inputVector);

public:

	//Default constructor
	NeuralNetwork(std::vector<int> _layerSizes);

	//Shows overall statistics of network
	void showStats();

	//Weight's vector getter
	std::vector<Matrix2D<double>> getWeights();

	//Weight's vector setter
	void setWeights(Matrix2D<double> &_weightsMatrix, int _layer);

	//Biases's vector getter
	std::vector<std::vector<double>> getBiases();

	//Biases's vector setter
	void setBiases(std::vector<std::vector<double>> &_biasesTable);

	//Function used to calculate a single network ouput(response) from a single input
	std::vector<double> predict(std::vector<double> _inputVector);

	//Algorithm used to implement the learning features of network, it is resposnisble for weights corection so that netwrok can learn
	std::pair< std::vector<std::vector<double>>, std::vector<Matrix2D<double>>> backpropagation(std::vector<double>& _inputVector,
		std::vector<int>& _inputVector2);

	//Function resposible for updating a single learning unit of network, mainly using backpropagation
	void updateMiniBatch(std::vector<std::pair<std::vector<double>, std::vector<int>>> miniBatch, double eta);

	//Function responsible for whole learning process, slowly updating and adjusting all nuerons to achive the best performable network
	void SGD(std::vector<std::pair<std::vector<double>, std::vector<int>>>& trainingData, int epochs, int miniBatchSize,
		double eta, std::vector<std::pair<std::vector<double>, std::vector<int>>>& testData);

	//TO_BE_DELETED_FURTHER, Function printing weight shapes
	void testPrintWeightShapes();

	//TO_BE_DELETED_FURTHER, Function printing weight matrixes
	void testPrintMatrixes();

	//TO_BE_DELETED_FURTHER, Function printing biases
	void testPrintBiases();
};