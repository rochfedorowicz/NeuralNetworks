#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <streambuf>

#include "Matrix2D.tpp"

std::vector<double> matrixByVectorMultiplication(Matrix2D<double>& _inputMatrix, std::vector<double>& _inputVector) {
	std::vector<double> newVector;
	for (int i = 0; i < _inputMatrix.getRowsNumber(); ++i) {
		double outcome = 0;
		for (int j = 0; j < _inputMatrix.getRow(i)->size(); ++j) {
			outcome += _inputMatrix.getRow(i)->at(j) * _inputVector.at(j);
		}
		newVector.push_back(outcome);
	}
	return newVector;
}

template <typename inputType> std::size_t getNumberOfMaxOfVector(std::vector<inputType>& _vec) {
	size_t maxValNr = 0;
	for (int i = 1; i < _vec.size(); ++i) {
		if (_vec.at(maxValNr) < _vec.at(i)) maxValNr = i;
	}
	return maxValNr;
}

std::vector<double> vectorToVectorAdding(std::vector<double> _inputVector, std::vector<double>& _inputVector2) {
	std::vector<double> newVector;
	for (int i = 0; i < _inputVector.size(); ++i) newVector.push_back(_inputVector.at(i) + _inputVector2.at(i));
	return newVector;
}

template <typename Type> std::vector<double> vectorFromVectorSubtrackting(std::vector<double> _inputVector, std::vector<Type> _inputVector2) {
	std::vector<double> newVector;
	for (int i = 0; i < _inputVector.size(); ++i) newVector.push_back(_inputVector.at(i) - _inputVector2.at(i));
	return newVector;
}

std::vector<double> vectorByVectorMultiplyingElementWise(std::vector<double> _inputVector, std::vector<double>& _inputVector2) {
	std::vector<double> newVector;
	for (int i = 0; i < _inputVector.size(); ++i) newVector.push_back(_inputVector.at(i) * _inputVector2.at(i));
	return newVector;
}

std::vector<double> vectorByValueMultiplying(std::vector<double> _inputVector, double value) {
	std::vector<double> newVector;
	for (int i = 0; i < _inputVector.size(); ++i) newVector.push_back(_inputVector.at(i) * value);
	return newVector;
}

Matrix2D<double> getMatrixFromVextorMultiplying(std::vector<double>& _inputVector, std::vector<double>& _inputVector2) {
	Matrix2D<double> newMatrix(_inputVector.size(), _inputVector2.size(), ModeOfMaterixInit::ZEROS);
	for (int i = 0; i < _inputVector.size(); ++i) {
		for (int j = 0; j < _inputVector2.size(); ++j) {
			newMatrix.getRow(i)->at(j) = _inputVector.at(i) * _inputVector2.at(j);
		}
	}
	return newMatrix;
}

class NeuralNetwork {
	std::vector<int> layerSizes;
	std::vector<std::pair<int, int>> weightShapes;
	std::vector<Matrix2D<double>> weights;
	std::vector<std::vector<double>> biases;

	double sygmoidFn(double val) {
		return 1.0 / (1.0 + std::exp(-1 * val));
	}

	double sygmoidPrimeFn(double val) {
		return sygmoidFn(val) * (1.0 - sygmoidFn(val));
	}

	void activation(std::vector<double>& _inputVector) {
		std::for_each(_inputVector.begin(), _inputVector.end(), [this](double& val) {val = this->sygmoidFn(val); });
	}

	void reversActivation(std::vector<double>& _inputVector) {
		std::for_each(_inputVector.begin(), _inputVector.end(), [this](double& val) {val = this->sygmoidPrimeFn(val); });
	}

public:

	NeuralNetwork(std::vector<int> _layerSizes) {

		//Added lately
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

	std::vector<Matrix2D<double>> getWeights() {
		return weights;
	}

	std::vector<double> predict(std::vector<double> _inputVector) {
		std::vector<double> outputVector = _inputVector;
		for (int i = 0; i < weights.size(); ++i) {
			outputVector = vectorToVectorAdding(matrixByVectorMultiplication(weights.at(i), outputVector), biases.at(i));
			activation(outputVector);
		}
		return outputVector;
	}

	std::pair< std::vector<std::vector<double>>, std::vector<Matrix2D<double>>> backpropagation(std::vector<double>& _inputVector,
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
			auto z = vectorToVectorAdding(matrixByVectorMultiplication(weights.at(i), activationVector), biases.at(i));
			zVectros.push_back(z);
			activation(z);
			activationVector = z;
			activationsVectors.push_back(activationVector);
		}
		reversActivation(zVectros.back());
		auto delta = vectorByVectorMultiplyingElementWise(vectorFromVectorSubtrackting<int>(activationsVectors.back(), _inputVector2),
			zVectros.back());
		newBiases.back() = delta;
		newWeights.back()= getMatrixFromVextorMultiplying(delta, activationsVectors.at(activationsVectors.size() - 2));
		//continue here
		for (int i = 2; i < layerSizes.size(); i++) {
			auto v = zVectros.at(zVectros.size() - i);
			reversActivation(v);
			auto tempMat = weights.at(weights.size() - i + 1).getTransposedMatrix();
			delta = vectorByVectorMultiplyingElementWise(matrixByVectorMultiplication(tempMat, delta), v);
			newBiases.at(newBiases.size() - i) = delta;
			newWeights.at(newWeights.size() - i) = getMatrixFromVextorMultiplying(delta,
				activationsVectors.at(activationsVectors.size() - i - 1));
		}
		return {newBiases, newWeights};
	}

	void updateMiniBatch(std::vector<std::pair<std::vector<double>, std::vector<int>>> miniBatch, double eta) {
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
		std::for_each(miniBatch.begin(), miniBatch.end(), [&](std::pair<std::vector<double>, std::vector<int>> &pair) {
			auto result = backpropagation(pair.first, pair.second);
			for (int i = 0; i < newBiases.size(); ++i) {
				newBiases.at(i) = vectorToVectorAdding(newBiases.at(i), result.first.at(i));
				newWeights.at(i) = newWeights.at(i) + result.second.at(i);
			}
			for (int i = 0; i < newBiases.size(); ++i) {
				weights.at(i) = weights.at(i) - newWeights.at(i) * (eta / miniBatch.size());
				biases.at(i) = vectorFromVectorSubtrackting<double>(biases.at(i),
					vectorByValueMultiplying(newBiases.at(i), eta / miniBatch.size()));
			}
		});
	}

	void SGD(std::vector<std::pair<std::vector<double>, std::vector<int>>>& trainingData, int epochs, int miniBatchSize,
		double eta, std::vector<std::pair<std::vector<double>, std::vector<int>>>& testData) {
		//auto newEta = eta;
		for (int i = 0; i < epochs; ++i) {
			//newEta = 2 * eta * (1 - prec / 100);
			std::random_shuffle(trainingData.begin(), trainingData.end());
			//system("PAUSE");
			//std::cout << "Data shuffeled\n";
			int n = trainingData.size();
			std::vector<std::vector<std::pair<std::vector<double>, std::vector<int>>>> miniBatches;
			//system("PAUSE");
			//std::cout << "Proceed to divide into minibatches\n";
			for (int j = 0; j < trainingData.size(); j += miniBatchSize) {
				miniBatches.push_back({ trainingData.begin() + j, trainingData.begin() + j + miniBatchSize - 1 >= trainingData.end() ? trainingData.end() : trainingData.begin() + j + miniBatchSize - 1 });
			}
			//system("PAUSE");
			//std::cout << "Divited into minibatches\n";
			for (int j = 0; j < miniBatches.size(); ++j) {
				updateMiniBatch(miniBatches.at(j), eta);
			//	std::cout << "Mini batch nr " << j + 1 << " out of " << miniBatches.size() << " updated\n";
			}
			//system("PAUSE");
			//std::cout << "All mini batches updated\n";
			std::cout << "Epoch " << i + 1 << " out of " << epochs << " completed.\n";
			std::size_t nrOfCorrectGuesses = 0;
			for (int j = 0; j < testData.size(); ++j) {
				std::vector<double> prediction = predict(testData.at(j).first);
				if (getNumberOfMaxOfVector<double>(prediction) == getNumberOfMaxOfVector<int>(testData.at(j).second)) ++nrOfCorrectGuesses;
			}
			auto prec = nrOfCorrectGuesses * 100.0 / testData.size();
			//testPrintMatrixes();
			std::cout << "\n" << prec << "% of precision\n";
		}
	}
	//tests

	void testPrintWeightShapes() {
		std::for_each(weightShapes.begin(), weightShapes.end(), [](std::pair<int, int>& wS) {std::cout << "(" << wS.first << ", " << wS.second << ") "; });
		std::cout << std::endl;
	}

	void testPrintMatrixes() {
		std::for_each(weights.begin(), weights.end(), [](Matrix2D<double>& mat) {mat.printMatrix(); std::cout << std::endl; });
	}

	void testPrintBiases() {
		std::for_each(biases.begin(), biases.end(), [](std::vector<double>& b) {std::for_each(b.begin(), b.end(), [](double& d) {std::cout << d << "\n"; }); std::cout << std::endl; });
	}
};

class Reader {

	enum class readingMode { INTEGER_PART, FRACTIONAL_PART };

public:

	std::vector<int> readToVectorOfInts(std::istream& _fs) {
		std::vector<int> readedVector;
		std::string tempLine;
		while (std::getline(_fs, tempLine))
		{
			if (!tempLine.empty()) {
				std::vector<int> integerPartDigits;
				for (int i = 0; i < tempLine.length(); ++i) {
					if (tempLine[i] >= 48 && tempLine[i] < 58) integerPartDigits.push_back(tempLine[i] - 48);
				}
				int tempVal = 0;
				int power = 1;
				for (auto it = integerPartDigits.rbegin(); it != integerPartDigits.rend(); ++it) {
					tempVal += power * *it;
					power *= 10;
				}
				readedVector.push_back(tempVal);
			}
		}
		return readedVector;
	}

	std::vector<double> readToVectorOfDoubles(std::istream& _fs) {
		std::vector<double> readedVector;
		std::string tempLine;
		while (std::getline(_fs, tempLine))
		{
			if (!tempLine.empty()) {
				std::vector<int> integerPartDigits, fractionalPartDigits;
				readingMode mode = readingMode::INTEGER_PART;
				for (int i = 0; i < tempLine.length(); ++i) {
					if (tempLine[i] >= 48 && tempLine[i] < 58) {
						if (mode == readingMode::INTEGER_PART) integerPartDigits.push_back(tempLine[i] - 48);
						if (mode == readingMode::FRACTIONAL_PART) fractionalPartDigits.push_back(tempLine[i] - 48);
					}
					if (tempLine[i] == '.') mode = readingMode::FRACTIONAL_PART;
				}
				double tempVal = 0;
				double power = 1;
				for (auto it = integerPartDigits.rbegin(); it != integerPartDigits.rend(); ++it) {
					tempVal += power * *it;
					power *= 10;
				}
				double power2 = 1.0 / 10;
				for (auto it = fractionalPartDigits.begin(); it != fractionalPartDigits.end(); ++it) {
					tempVal += power2 * *it;
					power2 *= 1.0 / 10;
				}
				readedVector.push_back(tempVal);
			}

		}
		return readedVector;
	}

	template <typename inputType> std::vector<std::vector<inputType>> getMatrixizedVecotr(std::vector<inputType> &_vector, std::size_t _nrOfElementsInARow) {
		std::vector<std::vector<inputType>> matrix;
		for (std::size_t i = 0, j = 0; i < _vector.size(); ++i, ++j) {
			if (j >= _nrOfElementsInARow) j = 0;
			if (j == 0) {
				matrix.push_back(std::vector<inputType>());
			}
			matrix.rbegin()->push_back(_vector.at(i));
		}
		return matrix;
	}

};

template <typename typeName> void writeDataByBytes(std::vector<std::vector<typeName>> dataToBeWritten, std::string fileName) {
	std::ofstream stream(fileName, std::ios::binary);
	typeName* buffer = new typeName;
	char* newBuffer = new char[sizeof(typeName)];
	std::for_each(dataToBeWritten.begin(), dataToBeWritten.end(), [&stream, &buffer, &newBuffer](std::vector<typeName> &vec) {
		for (int i = 0; i < vec.size(); ++i) {
			*buffer = vec.at(i);
			std::memcpy(newBuffer, buffer, sizeof(typeName));
			stream.write(newBuffer, sizeof(typeName));
		}
	});
	delete buffer;
	delete[] newBuffer;
	stream.close();
}

template <typename typeName> std::vector<typeName> readDataByBytes(std::string fileName) {
	std::ifstream stream(fileName, std::ios::binary);
	typeName tempVal;
	std::vector<typeName> tempVec;
	auto size = sizeof(typeName);
	char* buffer = new char[size];
	typeName* newBuffer = new typeName;
	while (!stream.eof()) {
		if (stream.read(buffer, size)) {
			std::memcpy(newBuffer, buffer, size);
			tempVec.push_back(*newBuffer);
		}
	};
	delete[] buffer;
	delete newBuffer;
	stream.close();
	return tempVec;
}

std::vector<std::pair<std::vector<double>, std::vector<int>>> getPairedTrainngData(std::vector<std::vector<double>> &inputData,
	std::vector<std::vector<int>> &outputData) {
	std::vector<std::pair<std::vector<double>, std::vector<int>>> traningSet;
	for (int i = 0; i < outputData.size(); ++i) {
		traningSet.push_back({
			inputData.at(i),
			outputData.at(i)});
	}
	return traningSet;
}

int main() {

	Reader reader;
	//std::ifstream file1("training_labels.txt");
//	auto time0 = std::chrono::high_resolution_clock().now();
//	//auto vec = reader.readToVectorOfInts(file1);
//	//auto matrix_output = reader.getMatrixizedVecotr<int>(vec, 10);
//	auto matrix_output = readDataByBytes<int>("outputData.byte", 10);
//	auto time1 = std::chrono::high_resolution_clock().now();
//	auto duration1 = std::chrono::duration_cast<std::chrono::seconds>(time1 - time0);
//	std::cout << "Labels data louded and it's taken " << duration1.count() << " seconds to perform.\n";
//	system("PAUSE");
//	//std::ifstream file2("training_images.txt");
//	time0 = std::chrono::high_resolution_clock().now();
////	auto vec2 = reader.readToVectorOfDoubles(file2);
////	auto matrix_input = reader.getMatrixizedVecotr<double>(vec2, 784);
//	auto matrix_input = readDataByBytes<double>("inputData.byte", 784);
//	time1 = std::chrono::high_resolution_clock().now();
//	duration1 = std::chrono::duration_cast<std::chrono::seconds>(time1 - time0);
//	std::cout << "Images data louded and it's taken " << duration1.count() << " seconds to perform.\n";
//	system("PAUSE");
	//writeDataByBytes<double>(matrix_input, "inputData.byte");
	//writeDataByBytes<int>(matrix_output, "outputData.byte");
	//for (int j = 0; j < 3; ++j) {
	//	NeuralNetwork network({ 784, 15, 10 });
	//	std::size_t nrOfCorrectGuesses = 0;
	//	for (int i = 0; i < 50000; ++i) {
	//		auto prediction = network.predict(matrix_input.at(i));
	//		if (getNumberOfMaxOfVector<double>(prediction) == getNumberOfMaxOfVector<int>(matrix_output.at(i))) ++nrOfCorrectGuesses;
	//	}
	//	std::cout << nrOfCorrectGuesses * 1.0 / 500 << "% of precision in " << j + 1 << " try\n";
	//}

	//std::ifstream file1("training_labels2.txt");
	//auto vec = reader.readToVectorOfInts(file1);
	//auto matrix_output = reader.getMatrixizedVecotr<int>(vec, 10);
	//file1.close();

	//system("PAUSE");
	//auto time0 = std::chrono::high_resolution_clock().now();
	//Matrix2D newMatrix(50000, 784, ModeOfMaterixInit::RANDOM);
	//auto time1 = std::chrono::high_resolution_clock().now();
	//auto duration1 = std::chrono::duration_cast<std::chrono::seconds>(time1 - time0);
	//std::cout << "Matrix generated and it's taken " << duration1.count() << " seconds to perform.\n";
	//system("PAUSE");

	//time0 = std::chrono::high_resolution_clock().now();
	//writeDataByBytes<double>(newMatrix.getTabel(), "test.byte");
	//time1 = std::chrono::high_resolution_clock().now();
	//duration1 = std::chrono::duration_cast<std::chrono::seconds>(time1 - time0);
	//std::cout << "Matrix written to binary file and it's taken " << duration1.count() << " seconds to perform.\n";
	//system("PAUSE");

	/*auto time0 = std::chrono::high_resolution_clock().now();
	auto vec = readDataByBytes<double>("inputData.byte");
	auto matrix_input = reader.getMatrixizedVecotr(vec, 784);
	auto time1 = std::chrono::high_resolution_clock().now();
	auto duration1 = std::chrono::duration_cast<std::chrono::seconds>(time1 - time0);
	std::cout << "Vector loaded from binary file and it's taken " << duration1.count() << " seconds to perform.\n";
	system("PAUSE");

	auto vec2 = readDataByBytes<int>("outputData.byte");
	auto matrix_output = reader.getMatrixizedVecotr(vec2, 10);

	NeuralNetwork network({ 784, 15, 10 });
	auto traningDataSet = getPairedTrainngData(matrix_input, matrix_output);
	std::vector<std::pair<std::vector<double>, std::vector<int>>>  trainingData = {traningDataSet.begin(),
		traningDataSet.begin() + 400};
	std::vector<std::pair<std::vector<double>, std::vector<int>>>  testData = { traningDataSet.begin() + 400,
		traningDataSet.begin() + 500 };
	system("PAUSE");*/

	/*NeuralNetwork network({2, 2, 1});
	std::random_device device;
	std::mt19937 generator(device());
	std::normal_distribution<double> distribution(0, 1);
	auto size = 5000;
	std::vector<std::pair<std::vector<double>, std::vector<int>>> data;
	for (int i = 0; i < size; ++i) {
		auto x = distribution(generator), y = distribution(generator);
		std::vector<double> vec = { x, y };
		std::for_each(vec.begin(), vec.end(), [&](double & val) {
			val = abs(val / sqrt(pow(x,2) + pow(y,2)));
		});
		std::vector<int> vec2;
		if (vec.at(0) / vec.at(1) < 1)  vec2.push_back(1);
		else vec2.push_back(0);
		std::pair<std::vector<double>, std::vector<int>> pair = { vec, vec2 };
		data.push_back(pair);
	}
	std::vector<std::pair<std::vector<double>, std::vector<int>>> trainingData = 
	{ data.begin(), data.begin() + 4000 }, testData = {data.begin() + 4000, data.end()};
		system("PAUSE");
	network.SGD(trainingData, 100, 100, 30, testData);
	system("PAUSE");
	while (true) {
		double newX, newY;
		std::cout << "Podaj X: ";
		std::cin >> newX;
		std::cout << " Podaj Y: ";
		std::cin >> newY;
		std::cout << "The response is:" << network.predict({ newX, newY }).at(0);
		if (network.predict({ newX, newY }).at(0) - (newX / newY < 1 ? 1 : 0) < 0.1)
			std::cout << "\nGuessed correctly\n";
		else std::cout << "\nFailed to guess\n";
	}*/

	auto data = readDataByBytes<int>("outputData.byte");
	auto data2 = readDataByBytes<double>("inputData.byte");
	auto mat1 = reader.getMatrixizedVecotr(data, 10);
	auto mat2 = reader.getMatrixizedVecotr(data2, 784);
	auto dataSet = getPairedTrainngData(mat2, mat1);
	std::random_shuffle(dataSet.begin(), dataSet.end());
	std::vector<std::pair<std::vector<double>, std::vector<int>>> trainingSet =
	{ dataSet.begin(), dataSet.begin() + 490 }, testSet = { dataSet.begin() + 490, dataSet.begin() + 500 };
	NeuralNetwork net({ 784, 45, 10 });
	net.SGD(trainingSet, 30, 10, 0.1, testSet);
	//int index = 1;
	//std::string name = "weightsLayer";
	//for (auto mat : net.getWeights()) {
	//	auto newName = name;
	//	name.push_back(index);
	//	name.append(".byte");
	//	writeDataByBytes<double>(mat.getTabel(), name);
	//	++index;
	//}
	return 0;
}
