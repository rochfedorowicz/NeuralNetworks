#pragma once
#include "Matrix2D.h"

//Multiplying vectors elemntwise
template <typename Type, typename Type2>
std::vector<Type> operator^(std::vector<Type> _firstVec, std::vector<Type2> _secondVec);

//Multiplying vector by scallar
template <typename Type, typename Type2>
std::vector<Type> operator*(std::vector<Type> _vec, Type2 _scallar);

//Multiplying vector, interpeted as column by another vector. Output is always matrix.
template <typename Type, typename Type2>
Matrix2D<Type> operator*(std::vector<Type> _firstVec, std::vector<Type2> _secondVec);

//Multiplying matrix by vector, interpeted as column. Output is always transposed column vector 
template <typename Type, typename Type2>
std::vector<Type> operator*(Matrix2D<Type> _matrix, std::vector<Type2> _vec);

//Multiplying matrix by scallar
template <typename Type, typename Type2>
Matrix2D<Type> operator>(Matrix2D<Type2> _matrix, Type _val);

//Adding vectors
template <typename Type, typename Type2>
std::vector<Type> operator+(std::vector<Type> _firstVec, std::vector<Type2>  _secondVec);

//Subtracting vectors
template <typename Type, typename Type2>
std::vector<Type> operator-(std::vector<Type> _firstVec, std::vector<Type2> _secondVec);

//Finds index of maximum value in vector
template <typename Type>
std::size_t getIndexOfMaximalValueInVector(std::vector<Type>& _vec);