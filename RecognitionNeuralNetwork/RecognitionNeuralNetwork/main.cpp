#include <fstream>
#include <sstream>
#include <thread>

#include "NeuralNetwork.h"
#include "Matrix2D.cpp"
#include "VectorOperands.cpp"

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

int main() {
	//auto dataMat = readDataByBytes<int>("outputData.byte", 10);
	//auto dataMat2 = readDataByBytes<double>("inputData.byte", 784);
	//auto dataSet = getPairedTrainngData(dataMat2, dataMat);
	//std::random_shuffle(dataSet.begin(), dataSet.end());
	//std::vector<std::pair<std::vector<double>, std::vector<int>>> trainingSet =
	//{ dataSet.begin(), dataSet.begin() + 250 }, testSet = { dataSet.begin() + 250, dataSet.begin() + 270 };
	NeuralNetwork net({ 784, 30, 10 });
	net.showStats();
	//auto w1 = readDataByBytes<double>("weightsLayer1.byte", 30);
	//auto w2 = readDataByBytes<double>("weightsLayer2.byte", 10);
	//auto b1 = readDataByBytes<double>("biases.byte", {30, 10});
	//net.setWeights(w1, 1);
	//net.setWeights(w2, 2);
	//net.setBiases(b1);

	//
	//int amount = 0, corr = 0;
	//for (auto& row : dataMat2.getTabel()) {
	//	auto pred = net.predict(row);
	//	if (getIndexOfMaximalValueInVector(pred) == getIndexOfMaximalValueInVector(*dataMat.getRowPtr(amount))) corr++;
	//	amount++;
	//}
	//std::cout << corr * 100.0 / amount << " % of correction...";
	//auto fn = [&]() {net.SGD(trainingSet, 3, 10, 0.1, testSet); };
	//std::thread th1(fn);
	//th1.join();


	//int index = 1;
	//std::string name = "weightsLayer";
	//for (	auto mat : net.getWeights()) {
	//	auto newName = name;
	//	newName.push_back(static_cast<char>(index + 48));
	//	newName.append(".byte");
	//	writeDataByBytes<double>(mat, newName);
	//	++index;
	//}
	//auto biasesTab = net.getBiases();
	//writeDataByBytes<double>(Matrix2D<double>(biasesTab), "biases.byte");

	return 0;
}
