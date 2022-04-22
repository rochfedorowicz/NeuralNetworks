#pragma once
#include "NeuralNetwork.h"
#include "Matrix2D.cpp"
#include "VectorOperands.cpp"

NeuralNetwork::NeuralNetwork(std::vector<int> _layerSizes) {

	layerSizes = _layerSizes;

	for (int i = 0; i < _layerSizes.size() - 1; ++i) {
		weightShapes.push_back(std::pair<int, int>(_layerSizes.at(i + 1), _layerSizes.at(i)));
		biases.push_back(std::vector<double>());
		for (int j = 0; j < _layerSizes.at(i + 1); ++j) {
			biases.back().push_back(0);
		}
	}

	std::for_each(weightShapes.begin(), weightShapes.end(), [&](std::pair<int, int>& wS) {
		auto tempMat = Matrix2D<double>(wS.first, wS.second, ModeOfMaterixInit::RANDOM);
		tempMat.normalize();
		weights.push_back(tempMat);
	});
}

double NeuralNetwork::sygmoidFn(double val) {
	return 1.0 / (1.0 + std::exp(-1 * val));
}
double NeuralNetwork::sygmoidPrimeFn(double val) {
	return sygmoidFn(val) * (1.0 - sygmoidFn(val));
}

void NeuralNetwork::activation(std::vector<double>& _inputVector) {
	std::for_each(_inputVector.begin(), _inputVector.end(), [this](double& val) {val = this->sygmoidFn(val); });
}

void NeuralNetwork::reversActivation(std::vector<double>& _inputVector) {
	std::for_each(_inputVector.begin(), _inputVector.end(), [this](double& val) {val = this->sygmoidPrimeFn(val); });
}

void NeuralNetwork::showStats() {
	for (auto& baias : biases) std::cout << baias.size() << " ";
	std::cout << std::endl;
	for (auto& weight : weights) std::cout << weight.getRowsNumber() << " x " << weight.getRowPtr(0)->size() << "\n";
}

std::vector<Matrix2D<double>> NeuralNetwork::getWeights() {
	return weights;
}

void NeuralNetwork::setWeights(std::vector<Matrix2D<double>>& _weightsVec) {
	weights = _weightsVec;
}

std::vector<std::vector<double>> NeuralNetwork::getBiases() {
	return biases;
}

void NeuralNetwork::setBiases(std::vector<std::vector<double>>& _biasesTable) {
	biases = _biasesTable;
}

std::vector<double> NeuralNetwork::predict(std::vector<double> _inputVector) {
	std::vector<double> outputVector = _inputVector;
	for (int i = 0; i < weights.size(); ++i) {
		outputVector = (weights.at(i) * outputVector) + biases.at(i);
		activation(outputVector);
	}
	return outputVector;
}

std::pair< std::vector<std::vector<double>>, std::vector<Matrix2D<double>>> NeuralNetwork::backpropagation(std::vector<double>& _inputVector,
	std::vector<int>& _inputVector2) {
	std::vector<std::vector<double>> newBiases;
	std::vector<Matrix2D<double>> newWeights;
	std::vector<double> activationVector = _inputVector;
	std::vector<std::vector<double>> activationsVectors, zVectros;
	activationsVectors.push_back(activationVector);
	for (int i = 0; i < layerSizes.size() - 1; ++i) {
		newBiases.push_back(std::vector<double>());
		for (int j = 0; j < layerSizes.at(i + 1); ++j) {
			newBiases.back().push_back(0);
		}
	}
	std::for_each(weightShapes.begin(), weightShapes.end(), [&](std::pair<int, int>& wS) {newWeights.push_back(Matrix2D<double>(wS.first, wS.second, ModeOfMaterixInit::ZEROS)); });
	for (int i = 0; i < weights.size(); ++i) {
		auto z = (weights.at(i) * activationVector) + biases.at(i);
		zVectros.push_back(z);
		activation(z);
		activationVector = z;
		activationsVectors.push_back(activationVector);
	}
	reversActivation(zVectros.back());
	auto delta = (activationsVectors.back() - _inputVector2) ^ zVectros.back();
	newBiases.back() = delta;
	newWeights.back() = delta * activationsVectors.at(activationsVectors.size() - 2);
	for (int i = 2; i < layerSizes.size(); i++) {
		auto v = zVectros.at(zVectros.size() - i);
		reversActivation(v);
		auto tempMat = weights.at(weights.size() - i + 1).getTransposedMatrix();
		delta = (tempMat * delta) ^ v;
		newBiases.at(newBiases.size() - i) = delta;
		newWeights.at(newWeights.size() - i) = delta * activationsVectors.at(activationsVectors.size() - i - 1);
	}
	return { newBiases, newWeights };
}

void NeuralNetwork::updateMiniBatch(std::vector<std::pair<std::vector<double>, std::vector<int>>> miniBatch, double eta) {
	std::vector<std::vector<double>> newBiases;
	std::vector<Matrix2D<double>> newWeights;
	for (int i = 0; i < layerSizes.size() - 1; ++i) {
		newBiases.push_back(std::vector<double>());
		for (int j = 0; j < layerSizes.at(i + 1); ++j) {
			newBiases.back().push_back(0);
		}
	}
	std::for_each(weightShapes.begin(), weightShapes.end(), [&](std::pair<int, int>& wS) {
		newWeights.push_back(Matrix2D<double>(wS.first, wS.second, ModeOfMaterixInit::ZEROS));
		});
	std::for_each(miniBatch.begin(), miniBatch.end(), [&](std::pair<std::vector<double>, std::vector<int>>& pair) {
		auto result = backpropagation(pair.first, pair.second);
		for (int i = 0; i < newBiases.size(); ++i) {
			newBiases.at(i) = newBiases.at(i) + result.first.at(i);
			newWeights.at(i) = newWeights.at(i) + result.second.at(i);
		}
		for (int i = 0; i < newBiases.size(); ++i) {
			weights.at(i) = weights.at(i) - (newWeights.at(i) > (eta / miniBatch.size()));
			biases.at(i) = biases.at(i) - (newBiases.at(i) * (eta / miniBatch.size()));
		}
	});
}

void NeuralNetwork::SGD(std::vector<std::pair<std::vector<double>, std::vector<int>>>&trainingData, int epochs, int miniBatchSize,
	double eta, std::vector<std::pair<std::vector<double>, std::vector<int>>>&testData) {
	for (int i = 0; i < epochs; ++i) {
		std::random_shuffle(trainingData.begin(), trainingData.end());
		std::vector<std::vector<std::pair<std::vector<double>, std::vector<int>>>> miniBatches;
		for (int j = 0; j < trainingData.size(); j += miniBatchSize) {
			miniBatches.push_back({ trainingData.begin() + j, trainingData.begin() + j + miniBatchSize - 1 >= trainingData.end() ? trainingData.end() : trainingData.begin() + j + miniBatchSize - 1 });
		}
		for (int j = 0; j < miniBatches.size(); ++j) {
			updateMiniBatch(miniBatches.at(j), eta);
		}
		std::cout << "Epoch " << i + 1 << " out of " << epochs << " completed.\n";
		std::size_t nrOfCorrectGuesses = 0;
		for (int j = 0; j < testData.size(); ++j) {
			std::vector<double> prediction = predict(testData.at(j).first);
			if (getIndexOfMaximalValueInVector(prediction) == getIndexOfMaximalValueInVector(testData.at(j).second))
				++nrOfCorrectGuesses;
		}
		auto prec = nrOfCorrectGuesses * 100.0 / testData.size();
		std::cout << "\n" << prec << "% of precision\n";
	}
}

void NeuralNetwork::testPrintWeightShapes() {
	std::for_each(weightShapes.begin(), weightShapes.end(), [](std::pair<int, int>& wS) {std::cout << "(" << wS.first << ", " << wS.second << ") "; });
	std::cout << std::endl;
}

void NeuralNetwork::testPrintMatrixes() {
	std::for_each(weights.begin(), weights.end(), [](Matrix2D<double>& mat) {mat.printMatrix(); std::cout << std::endl; });
}

void NeuralNetwork::testPrintBiases() {
	std::for_each(biases.begin(), biases.end(), [](std::vector<double>& b) {std::for_each(b.begin(), b.end(), [](double& d) {std::cout << d << "\n"; }); std::cout << std::endl; });
}