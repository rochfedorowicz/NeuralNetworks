#include "VectorOperands.h"

template <typename Type, typename Type2>
std::vector<Type> operator^(std::vector<Type> _firstVec, std::vector<Type2> _secondVec) {
	if (_firstVec.size() == _secondVec.size()) {
		std::vector<Type> newVec;
		for (int i = 0; i < _firstVec.size(); ++i) {
			newVec.push_back(_firstVec.at(i) * _secondVec.at(i));
		}
		return newVec;
	}
	else throw "Exception - attempt to mutiply vectors of different sizes";
}

template <typename Type, typename Type2>
std::vector<Type> operator*(std::vector<Type> _vec, Type2 _scallar) {
	std::vector<Type> newVec;
	for (int i = 0; i < _vec.size(); ++i) {
		newVec.push_back(_vec.at(i) * _scallar);
	}
	return newVec;
}

template <typename Type, typename Type2>
Matrix2D<Type> operator*(std::vector<Type> _firstVec, std::vector<Type2> _secondVec) {
	Matrix2D<Type> newMatrix(_firstVec.size(), _secondVec.size(), ModeOfMaterixInit::ZEROS);
	for (int i = 0; i < _firstVec.size(); ++i) {
		for (int j = 0; j < _secondVec.size(); ++j) {
			newMatrix.getRowPtr(i)->at(j) = _firstVec.at(i) * _secondVec.at(j);
		}
	}
	return newMatrix;
}

template <typename Type, typename Type2>
std::vector<Type> operator*(Matrix2D<Type> _matrix, std::vector<Type2> _vec) {
	std::vector<double> newVector;
	for (int i = 0; i < _matrix.getRowsNumber(); ++i) {
		double outcome = 0;
		for (int j = 0; j < _matrix.getRowPtr(i)->size(); ++j) {
			outcome += _matrix.getRowPtr(i)->at(j) * _vec.at(j);
		}
		newVector.push_back(outcome);
	}
	return newVector;
}

template <typename Type, typename Type2>
Matrix2D<Type> operator>(Matrix2D<Type2> _matrix, Type _val) {
	Matrix2D<Type> newMatrix(_matrix.getRowsNumber(), _matrix.getRowPtr(0)->size(), ModeOfMaterixInit::ZEROS);
	for (int i = 0; i < _matrix.getRowsNumber(); ++i) {
		for (int j = 0; j < _matrix.getRowPtr(0)->size(); ++j) {
			newMatrix.getRowPtr(i)->at(j) = _matrix.getRowPtr(i)->at(j) * _val;
		}
	}
	return newMatrix;
}

template <typename Type, typename Type2>
std::vector<Type> operator+(std::vector<Type> _firstVec, std::vector<Type2> _secondVec) {
	if (_firstVec.size() == _secondVec.size()) {
		std::vector<Type> newVec;
		for (int i = 0; i < _firstVec.size(); ++i) {
			newVec.push_back(_firstVec.at(i) + _secondVec.at(i));
		}
		return newVec;
	}
	else throw "Exception - attempt to add vectors of different sizes";
}

template <typename Type, typename Type2> 
std::vector<Type> operator-(std::vector<Type> _firstVec, std::vector<Type2> _secondVec) {
	if (_firstVec.size() == _secondVec.size()) {
		std::vector<Type> newVec;
		for (int i = 0; i < _firstVec.size(); ++i) {
			newVec.push_back(_firstVec.at(i) - _secondVec.at(i));
		}
		return newVec;
	}
	else throw "Exception - attempt to subtract vectors of different sizes";
}

template <typename Type>
std::size_t getIndexOfMaximalValueInVector(std::vector<Type>& _vec) {
	size_t maxValNr = 0;
	for (int i = 1; i < _vec.size(); ++i) {
		if (_vec.at(maxValNr) < _vec.at(i)) maxValNr = i;
	}
	return maxValNr;
}