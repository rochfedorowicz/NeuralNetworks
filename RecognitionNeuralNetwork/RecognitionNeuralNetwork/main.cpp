#include <fstream>
#include <sstream>

#include "NeuralNetwork.h"

template <typename Type> std::vector<std::vector<Type>> getMatrixizedVecotr(std::vector<Type>& _vector, std::size_t _nrOfElementsInARow) {
	std::vector<std::vector<Type>> matrix;
	for (std::size_t i = 0, j = 0; i < _vector.size(); ++i, ++j) {
		if (j >= _nrOfElementsInARow) j = 0;
		if (j == 0) {
			matrix.push_back(std::vector<Type>());
		}
		matrix.rbegin()->push_back(_vector.at(i));
	}
	return matrix;
}

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
	auto data = readDataByBytes<int>("outputData.byte");
	auto data2 = readDataByBytes<double>("inputData.byte");
	auto mat1 = getMatrixizedVecotr(data, 10);
	auto mat2 = getMatrixizedVecotr(data2, 784);
	auto dataSet = getPairedTrainngData(mat2, mat1);
	std::random_shuffle(dataSet.begin(), dataSet.end());
	std::vector<std::pair<std::vector<double>, std::vector<int>>> trainingSet =
	{ dataSet.begin(), dataSet.begin() + 10000 }, testSet = { dataSet.begin() + 10000, dataSet.begin() + 11000 };
	NeuralNetwork net({ 784, 15, 10 });
	net.SGD(trainingSet, 30, 100, 0.3, testSet);

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
