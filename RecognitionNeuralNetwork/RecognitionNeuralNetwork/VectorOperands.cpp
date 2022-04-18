#include "VectorOperands.h"

template <typename Type>
Matrix2D<Type> getTransposedVector(std::vector<Type>& _vec) {
	Matrix2D<Type> newMatrix(_vec);
	return newMatrix.getTransposedMatrix();
}

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
std::vector<Type> operator*(std::vector<Type> _firstVec, Type2 _scallar) {
	std::vector<Type> newVec;
	for (int i = 0; i < _firstVec.size(); ++i) {
		newVec.push_back(_firstVec.at(i) * _scallar);
	}
	return newVec;
}

template <typename Type, typename Type2>
Matrix2D<Type> operator*(std::vector<Type> _vec, Matrix2D<Type2> _matrix) {
	Matrix2D<Type> newMat(_vec);
	return newMat * _matrix;
}

template <typename Type, typename Type2>
Matrix2D<Type> operator*(Matrix2D<Type> _matrix, std::vector<Type2> _vec) {
	Matrix2D<Type> newMat(_vec);
	return _matrix * newMat;
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