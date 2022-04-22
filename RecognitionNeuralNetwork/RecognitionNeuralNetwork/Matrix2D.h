#pragma once
#include<vector>
#include <random>
#include <iostream>
#include <algorithm>

//Enum for restricting options of initialization
enum class ModeOfMaterixInit { ZEROS, RANDOM };

//Class used to carry mathemtical operations in NeuralNetworks
template <typename Type> class Matrix2D {

//Matrix2D is implemented as a vecotr of vector of certain type
std::vector<std::vector<Type>> matrixTable;

public:
	//Default constructor
	Matrix2D(int _nrows, int _ncols, ModeOfMaterixInit _mode);

	//Table's constructor
	Matrix2D(std::vector<std::vector<Type>>& _matrixTable);

	//Getter for number of rows
	int getRowsNumber();

	//Getter for pointer to certain row
	std::vector<Type>* getRowPtr(int _rowNumber);

	//Performing normalization
	void normalize();

	//Getter for transposed matrix
	Matrix2D<Type> getTransposedMatrix();

	//Adding operator overloaded
	template <typename Type2> Matrix2D<Type> operator+(Matrix2D<Type2> _matrixToBeAdded);

	//Subtracting operator overloaded
	template <typename Type2> Matrix2D<Type> operator-(Matrix2D<Type2> _matrixToSubstracted);

	//Multiplying operator overloaded (by another matrix)
	template <typename Type2> Matrix2D<Type> operator*(Matrix2D<Type2> _matrixToByMultiplyed);

	//Printing matrix to standard cout
	void printMatrix();

	//Gettor for table of matrix (structre of vector of vector)
	std::vector<std::vector<Type>> getTabel();
};

