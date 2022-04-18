#pragma once
#include "Matrix2D.h"

//Transposing vector
template <typename Type>
Matrix2D<Type> getTransposedVector(std::vector<Type>& _vec);

//Multiplying vectors elemntwise
template <typename Type, typename Type2>
std::vector<Type> operator^(std::vector<Type> _firstVec, std::vector<Type2> _secondVec);

//Multiplying vectors by scallar
template <typename Type, typename Type2>
std::vector<Type> operator*(std::vector<Type> _firstVec, Type2 _scallar);

//Multiplying vectors by matrix
template <typename Type, typename Type2>
Matrix2D<Type> operator*(std::vector<Type> _vec, Matrix2D<Type2> _matrix);

//Multiplying matrix by vector
template <typename Type, typename Type2>
Matrix2D<Type> operator*(Matrix2D<Type> _matrix, std::vector<Type2> _vec);

//Multiplying matrix by scallar
template <typename Type, typename Type2>
Matrix2D<Type> operator>(Matrix2D<Type2> _matrix, Type _val);

//Adding vectors
template <typename Type, typename Type2>
std::vector<Type> operator+(std::vector<Type> _firstVec, std::vector<Type2>  _secondVec);

//Subtracting vectors
template <typename Type, typename Type2>
std::vector<Type> operator-(std::vector<Type> _firstVec, std::vector<Type2> _secondVec);