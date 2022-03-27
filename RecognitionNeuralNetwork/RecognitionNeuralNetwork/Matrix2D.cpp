#pragma once
#include "Matrix2D.h";

template <typename Type>
Matrix2D<Type>::Matrix2D(int _nrows, int _ncols, ModeOfMaterixInit _mode) {
	if (_mode == ModeOfMaterixInit::ZEROS) {
		for (int i = 0; i < _nrows; ++i) {
			matrixTable.push_back(std::vector<Type>());
			for (int j = 0; j < _ncols; ++j) {
				matrixTable.at(i).push_back(static_cast<Type>(0));
			}
		}
	}

	if (_mode == ModeOfMaterixInit::RANDOM) {
		std::random_device device;
		std::mt19937 generator(device());
		std::normal_distribution<Type> distribution(0, 1);
		for (int i = 0; i < _nrows; ++i) {
			matrixTable.push_back(std::vector<Type>());
			for (int j = 0; j < _ncols; ++j) {
				matrixTable.at(i).push_back(distribution(generator));
			}
		}
	}
}

template <typename Type>
Matrix2D<Type>::Matrix2D(std::vector<std::vector<Type>>& _matrixTable) {
	this->matrixTable = _matrixTable;
}

template <typename Type>
Matrix2D<Type>::Matrix2D(std::vector<Type>& _vectorToConvert) {
	std::vector<std::vector<Type>> newTable({_vectorToConvert});
	this->matrixTable = newTable;
}

template <typename Type> 
int Matrix2D<Type>::getRowsNumber() {
	return matrixTable.size();
}

template <typename Type>
std::vector<Type>* Matrix2D<Type>::getRow(int _rowNumber) {
	return  &matrixTable.at(_rowNumber);
}

template <typename Type> 
void Matrix2D<Type>::normalize() {
	for (int i = 0; i < matrixTable.size(); ++i) {
		double quadraticsum = 0;
		std::for_each(matrixTable.at(i).begin(), matrixTable.at(i).end(), [&quadraticsum](double& val) {
			quadraticsum += pow(val, 2);
		});
		for (int j = 0; j < matrixTable.at(i).size(); ++j) {
			matrixTable.at(i).at(j) = abs(matrixTable.at(i).at(j) / sqrt(quadraticsum));
		}
	}
}

template <typename Type>
Matrix2D<Type> Matrix2D<Type>::getTransposedMatrix() {
	Matrix2D newMatrix(matrixTable.at(0).size(), matrixTable.size(), ModeOfMaterixInit::ZEROS);
	for (int i = 0; i < matrixTable.at(0).size(); ++i) {
		for (int j = 0; j < matrixTable.size(); ++j) {
			newMatrix.getRow(i)->at(j) = matrixTable.at(j).at(i);
		}
	}
	return newMatrix;
}

template <typename Type> template <typename Type2>
Matrix2D<Type> Matrix2D<Type>::operator+(Matrix2D<Type2> _matrixToBeAdded) {
	if (this->getRowsNumber() == _matrixToBeAdded.getRowsNumber()
		&& this->getRow(0)->size() == _matrixToBeAdded.getRow(0)->size()) {
		Matrix2D newMatrix(matrixTable.size(), matrixTable.at(0).size(), ModeOfMaterixInit::ZEROS);
		for (int i = 0; i < matrixTable.size(); ++i) {
			for (int j = 0; j < matrixTable.at(i).size(); ++j) {
				newMatrix.getRow(i)->at(j) = this->getRow(i)->at(j) + _matrixToBeAdded.getRow(i)->at(j);
			}
		}
		return newMatrix;
	}
	else throw "Exception - attempt to add matrixes of different dimensions";
}

template <typename Type> template <typename Type2>
Matrix2D<Type> Matrix2D<Type>::operator-(Matrix2D<Type2> matrixToSubstracted) {
	if (this->getRowsNumber() == matrixToSubstracted.getRowsNumber()
		&& this->getRow(0)->size() == matrixToSubstracted.getRow(0)->size()) {
		Matrix2D newMatrix(matrixTable.size(), matrixTable.at(0).size(), ModeOfMaterixInit::ZEROS);
		for (int i = 0; i < matrixTable.size(); ++i) {
			for (int j = 0; j < matrixTable.at(i).size(); ++j) {
				newMatrix.getRow(i)->at(j) = this->getRow(i)->at(j) - matrixToSubstracted.getRow(i)->at(j);
			}
		}
		return newMatrix;
	}
	else throw "Exception - attempt to substruct matrixes of different dimensions";
}

template <typename Type> template <typename Type2>
Matrix2D<Type> Matrix2D<Type>::operator*(Matrix2D<Type2> _matrixToByMultiplyed) {
	if (this->getRow(0)->size() == _matrixToByMultiplyed.getRowsNumber()) {
		Matrix2D newMatrix(this->getRowsNumber(), _matrixToByMultiplyed.getRow(0)->size(), ModeOfMaterixInit::ZEROS);
		for (int i = 0; i < matrixTable.size(); ++i) {
			for (int j = 0; j < _matrixToByMultiplyed.getRow(0)->size(); ++j) {
				Type sum = 0;
				for (int k = 0; k < _matrixToByMultiplyed.getRowsNumber(); ++k) {
					sum += matrixTable.at(i).at(k) * _matrixToByMultiplyed.getRow(k)->at(j);
				}
				newMatrix.getRow(i)->at(j) = sum;
			}
		}
		return newMatrix;
	}
	else throw "Exception - attempt to multiply matrixes of not matching dimensions";
}

template <typename Type>
void Matrix2D<Type>::printMatrix() {
	for (int i = 0; i < matrixTable.size(); ++i) {
		for (int j = 0; j < matrixTable.at(i).size(); ++j) {
			std::cout << matrixTable.at(i).at(j) << " ";
		}
		std::cout << std::endl;
	}
}

template <typename Type>
std::vector<std::vector<Type>> Matrix2D<Type>::getTabel() {
	return matrixTable;
}