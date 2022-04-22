#include <fstream>
#include <sstream>
#include <thread>

#include "NeuralNetwork.h"
#include "Matrix2D.cpp"
#include "VectorOperands.cpp"

//TODO Create reader/writer class (IO handler) to make it code more clear
//TODO Implement a way to read PNG's

template <typename Type> Matrix2D<Type> chunkVector(std::vector<Type>& _vector, std::size_t _nrOfElementsInARow) {
	std::vector<std::vector<Type>> matrixV;
	for (std::size_t i = 0, j = 0; i < _vector.size(); ++i, ++j) {
		if (j >= _nrOfElementsInARow) j = 0;
		if (j == 0) {
			matrixV.push_back(std::vector<Type>());
		}
		matrixV.rbegin()->push_back(_vector.at(i));
	}
	return Matrix2D<Type>(matrixV);
}

template <typename Type> std::vector<std::vector<Type>> chunkVector(std::vector<Type>& _vector, std::vector<std::size_t> _vecOfElementsInRows) {
	std::vector<std::vector<Type>> matrixV;
	size_t rowNr = 0;
	for (std::size_t i = 0, j = 0; i < _vector.size(); ++i, ++j) {
		if (j >= _vecOfElementsInRows.at(rowNr)) {
			j = 0;
			rowNr++;
		}
		if (j == 0) {
			matrixV.push_back(std::vector<Type>());
		}
		matrixV.rbegin()->push_back(_vector.at(i));
	}
	return matrixV;
}

template <typename Type> void writeDataByBytes(Matrix2D<Type> dataToBeWritten, std::string fileName) {
	std::vector<std::vector<Type>> tableOfData = dataToBeWritten.getTabel();
	std::ofstream stream(fileName, std::ios::binary);
	Type* buffer = new Type;
	char* newBuffer = new char[sizeof(Type)];
	std::for_each(tableOfData.begin(), tableOfData.end(), [&stream, &buffer, &newBuffer](std::vector<Type> &vec) {
		for (int i = 0; i < vec.size(); ++i) {
			*buffer = vec.at(i);
			std::memcpy(newBuffer, buffer, sizeof(Type));
			stream.write(newBuffer, sizeof(Type));
		}
	});
	delete buffer;
	delete[] newBuffer;
	stream.close();
}

template <typename Type> void writeDataByBytes(std::vector<std::vector<Type>> dataToBeWritten, std::string fileName) {
	std::ofstream stream(fileName, std::ios::binary);
	Type* buffer = new Type;
	char* newBuffer = new char[sizeof(Type)];
	std::for_each(dataToBeWritten.begin(), dataToBeWritten.end(), [&stream, &buffer, &newBuffer](std::vector<Type>& vec) {
		for (int i = 0; i < vec.size(); ++i) {
			*buffer = vec.at(i);
			std::memcpy(newBuffer, buffer, sizeof(Type));
			stream.write(newBuffer, sizeof(Type));
		}
		});
	delete buffer;
	delete[] newBuffer;
	stream.close();
}

template <typename Type> Matrix2D<Type> readDataByBytes(std::string fileName, std::size_t _nrOfElementsInARow) {
	std::ifstream stream(fileName, std::ios::binary);
	Type tempVal;
	std::vector<Type> tempVec;
	auto size = sizeof(Type);
	char* buffer = new char[size];
	Type* newBuffer = new Type;
	while (!stream.eof()) {
		if (stream.read(buffer, size)) {
			std::memcpy(newBuffer, buffer, size);
			tempVec.push_back(*newBuffer);
		}
	};
	delete[] buffer;
	delete newBuffer;
	stream.close();
	return chunkVector(tempVec, _nrOfElementsInARow);
}

template <typename Type> std::vector<std::vector<Type>> readDataByBytes(std::string fileName, std::vector<std::size_t> _vecOfElementsInRows) {
	std::ifstream stream(fileName, std::ios::binary);
	Type tempVal;
	std::vector<Type> tempVec;
	auto size = sizeof(Type);
	char* buffer = new char[size];
	Type* newBuffer = new Type;
	while (!stream.eof()) {
		if (stream.read(buffer, size)) {
			std::memcpy(newBuffer, buffer, size);
			tempVec.push_back(*newBuffer);
		}
	};
	delete[] buffer;
	delete newBuffer;
	stream.close();
	return chunkVector(tempVec, _vecOfElementsInRows);
}

std::vector<std::pair<std::vector<double>, std::vector<int>>> getPairedTrainngData(Matrix2D<double> & _inputMatrix,
	Matrix2D<int> & _outputMatrix) {
	auto inputData = _inputMatrix.getTabel();
	auto outputData = _outputMatrix.getTabel();
	std::vector<std::pair<std::vector<double>, std::vector<int>>> traningSet;
	for (int i = 0; i < outputData.size(); ++i) {
		traningSet.push_back({
			inputData.at(i),
			outputData.at(i)});
	}
	return traningSet;
}

//TO_BE_DELETED_FURTHER, Function writing weights and biases down to byte files
void serializeWeightsAndBiases(NeuralNetwork &_net, std::string _nameOfNetwork) {
	int index = 1;
	std::string name, name2;
	name = name2 = _nameOfNetwork;
	name.append(" - weightsLayer");
	name2.append(" - biases.byte");
	for (auto mat : _net.getWeights()) {
		auto newName = name;
		newName.push_back(static_cast<char>(index + 48));
		newName.append(".byte");
		writeDataByBytes<double>(mat, newName);
		++index;
	}
	auto biasesTab = _net.getBiases();
	writeDataByBytes<double>(biasesTab, name2);
}

//TO_BE_DELETED_FURTHER, Function evaluating dataSet over the given net
void evaluateDataSet(std::vector<std::pair<std::vector<double>, std::vector<int>>> &_dataSet, NeuralNetwork& _net) {
	int amount = 0, corr = 0;
	for (auto& data : _dataSet) {
		auto pred = _net.predict(data.first);
		if (getIndexOfMaximalValueInVector(pred) == getIndexOfMaximalValueInVector(data.second)) corr++;
		amount++;
	}
	std::cout << std::endl << corr * 100.0 / amount << " % of correction...\n";
}

//TO_BE_DELETED_FURTHER, Function reading biases and weights from byte files
std::pair<std::vector<Matrix2D<double>>, std::vector<std::vector<double>>> deserializeWeightsAndBiases(std::string _nameOfNetwork,
	std::vector<int> _layerSizes) {
	std::vector<Matrix2D<double>> rWeights;
	std::vector<std::vector<double>> rBiases;
	for (int i = 0; i < _layerSizes.size() - 1; ++i) {
		auto newName = _nameOfNetwork;
		newName.append(" - weightsLayer");
		newName.push_back(static_cast<char>(i + 49));
		newName.append(".byte");
		rWeights.push_back(readDataByBytes<double>(newName, _layerSizes.at(i)));
	}
	auto newName2 = _nameOfNetwork;
	newName2.append(" - biases.byte");
	rBiases = readDataByBytes<double>(newName2, { _layerSizes.begin() + 1, _layerSizes.end() });
	return {rWeights, rBiases};
}

int main() {
	auto dataMat = readDataByBytes<int>("outputData.byte", 10);
	auto dataMat2 = readDataByBytes<double>("inputData.byte", 784);
	auto dataSet = getPairedTrainngData(dataMat2, dataMat);
	std::random_shuffle(dataSet.begin(), dataSet.end());
	std::vector<std::pair<std::vector<double>, std::vector<int>>> trainingSet =
	{ dataSet.begin(), dataSet.begin() + 250 }, testSet = { dataSet.begin() + 250, dataSet.begin() + 270 };
	std::vector<int> layerSizes = { 784, 30, 10 };
	NeuralNetwork net(layerSizes);

	std::string netName = "NN1";
	evaluateDataSet(testSet, net);
	serializeWeightsAndBiases(net, netName);
	auto readedData = deserializeWeightsAndBiases(netName, layerSizes);
	net.setWeights(readedData.first);
	net.setBiases(readedData.second);
	evaluateDataSet(testSet, net);

	//auto fn = [&]() {net.SGD(trainingSet, 3, 10, 0.1, testSet); };
	//std::thread th1(fn);
	//th1.join();

	return 0;
}
