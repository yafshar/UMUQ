#ifndef UMUQ_PYTHON_H
#define UMUQ_PYTHON_H
#ifdef HAVE_PYTHON

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif // _POSIX_C_SOURCE
#ifndef _AIX
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif // _XOPEN_SOURCE
#endif // _AIX

// Include Python.h before any standard headers are included
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// To avoid the compiler warning
#ifdef NPY_NO_DEPRECATED_API
#undef NPY_NO_DEPRECATED_API
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "core/core.hpp"
#include "datatype/npydatatype.hpp"
#include "misc/arraywrapper.hpp"

#if PYTHON_MAJOR_VERSION >= 3
#ifdef PyString_FromString
#undef PyString_FromString
#endif // PyString_FromString
#ifdef PyString_AsString
#undef PyString_AsString
#endif // PyString_AsString
#ifdef PyInt_FromLong
#undef PyInt_FromLong
#endif // PyInt_FromLong
#ifdef PyInt_FromString
#undef PyInt_FromString
#endif // PyInt_FromString
#define PyString_FromString PyUnicode_FromString
#define PyString_AsString PyUnicode_AsUTF8
#define PyInt_FromLong PyLong_FromLong
#define PyInt_FromString PyLong_FromString
#endif // PYTHON_MAJOR_VERSION >= 3

#include <complex>
#include <cstddef>

#include <string>
#include <vector>
#include <map>
#include <utility>

namespace umuq
{

/*! \defgroup Python_Module Python module
 * This is the python module of %UMUQ providing all necessary classes to embed Python.
 */

/*! \namespace umuq::python
 * \ingroup Python_Module
 *
 * \brief It is a minor companion to embed the Python Interpreter for all the
 * functionality one might need in %UMUQ.
 *
 */
namespace python
{

/*!
 * \file interface/python.hpp
 * \brief This module contains functions that allows to embed the Python Interpreter
 *
 * The python Module contains additions, adaptations and modifications to the
 * original c++ source codes [wrappy](https://github.com/lava/wrappy) made
 * available under the GPLv2 LICENSE:
 *
 * \verbatim
 * The general approach of providing a simple C++ API for utilizing python code.
 * Copyright (C) 2014 Benno Evers
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 * \endverbatim
 */
} // namespace python
} // namespace umuq

namespace umuq
{

/*! \namespace umuq::python
 * \ingroup Python_Module
 *
 * \brief It contains several approaches to make it easy to call out to python code from C++
 *
 * It contains several common approaches to make it easy to call out to python code from C++
 *
 */
namespace python
{
/*!
 * \ingroup Python_Module
 *
 * \brief Initialize python interpreter.
 *
 */
__attribute__((constructor)) void pythonInitialize()
{
  if (!Py_IsInitialized())
  {
// optional but recommended
// The name is used to find Python run-time libraries relative to the interpreter executable
#if PYTHON_MAJOR_VERSION >= 3
    wchar_t name[] = L"PYTHON_BIN";
#else
    char name[] = "PYTHON_BIN";
#endif

    // Pass name to the Python
    Py_SetProgramName(name);

    // Initialize python interpreter. Required.
    Py_Initialize();
  }

  if (PyArray_API == NULL)
  {
    // Initialize numpy
#if PYTHON_MAJOR_VERSION >= 3
    _import_array();
#else
    import_array();
#endif
  }
}

/*!
 * \ingroup Python_Module
 *
 * \brief Destruct python interpreter.
 *
 */
__attribute__((destructor)) void pythonFinalize()
{
  if (Py_IsInitialized())
  {
    Py_Finalize();
  }
}
} // namespace python
} // namespace umuq

namespace umuq
{
namespace python
{

/*!
 * \ingroup Python_Module
 *
 * \brief New type
 *
 */
using PyObjectMapChar = std::map<char const *, PyObject *>;
using PyObjectPairString = std::pair<std::string, PyObject *>;
using PyObjectPairChar = std::pair<char const *, PyObject *>;

using PyObjectVector = std::vector<PyObject *>;
using PyObjectPairStringVector = std::vector<PyObjectPairString>;
using PyObjectPairCharVector = std::vector<PyObjectPairChar>;

/*!
 * \brief Construct a Python object
 *
 * \param data Input data
 * \return PyObject*
 */
inline PyObject *PyObjectConstruct(int const data)
{
  return PyInt_FromLong(data);
}

inline PyObject *PyObjectConstruct(std::size_t const data)
{
  return PyLong_FromSize_t(data);
}

inline PyObject *PyObjectConstruct(float const data)
{
  return PyFloat_FromDouble(data);
}

inline PyObject *PyObjectConstruct(double const data)
{
  return PyFloat_FromDouble(data);
}

inline PyObject *PyObjectConstruct(std::string const &data)
{
  return PyString_FromString(data.c_str());
}

inline PyObject *PyObjectConstruc(PyObjectVector const &data)
{
  Py_ssize_t lSize = static_cast<Py_ssize_t>(data.size());
  // Return a new list of length len on success, or NULL on failure.
  PyObject *list = PyList_New(lSize); // Return value: New reference.
  for (Py_ssize_t index = 0; index < lSize; ++index)
  {
    PyObject *item = data[index];
    // PyList_SetItem steals a reference
    Py_XINCREF(item);
    // Set the item at index index in list to item.
    PyList_SetItem(list, index, item);
  }
  return list;
}

inline PyObject *PyObjectConstruct(PyObject *data)
{
  return data;
}

/*!
 * \ingroup Python_Module
 *
 * \brief A function pointer
 */
using PyObjectFunctionPointer = PyObject *(*)(PyObjectVector const &pyObjectVector, PyObjectMapChar const &pyObjectMapChar);

/*!
 * \ingroup Python_Module
 *
 * \brief A function pointer
 *
 */
using PyObjectFunctionPointerP = PyObject *(*)(void *PointerFunction, PyObjectVector const &pyObjectVector, PyObjectMapChar const &pyObjectMapChar);

/*!
 * \brief Evaluate the chain of dot-operators that leads from the module to
 *        the function.
 *
 * \param module Object to retrive the function attribute from it.
 * \param moduleName Module name
 *
 * \return PyObject*
 */
PyObject *PyGetFunction(PyObject *module, std::string const &moduleName)
{
  PyObject *object = module;
  std::size_t find_dot = 0;
  while (find_dot != std::string::npos)
  {
    std::size_t const next_dot = moduleName.find('.', find_dot + 1);
    std::string const attr = moduleName.substr(find_dot + 1, next_dot - (find_dot + 1));
    // Retrieve an attribute named attr from object \c object.
    object = PyObject_GetAttrString(object, attr.c_str()); // Return value: New reference.
    find_dot = next_dot;
  }
  return object;
}

/*!
 * \brief Call a function
 *
 * \param function A callable function object
 * \param pyObjectVector Vector of variable-length argument list
 * \param pyObjectPairStringVector Vector of keyworded, variable-length argument list
 *
 * \return PyObject*
 *
 * \note
 * Doesn't perform checks on the return value (input is still checked)
 */
PyObject *PyCallFunctionObject(PyObject *function,
                               PyObjectVector const &pyObjectVector,
                               PyObjectPairStringVector const &pyObjectPairStringVector)
{
  // Determine if the function is callable.
  // Return 1 if the function is callable and 0 otherwise. (always succeeds.)
  if (!PyCallable_Check(function))
  {
    UMUQFAILRETURNNULL("The input function isn't callable.");
  }

  // Build tuple
  Py_ssize_t const len = static_cast<Py_ssize_t>(pyObjectVector.size());

  // Return a new tuple object of size len, or NULL on failure.
  PyObject *tuple = PyTuple_New(len); // Return value: New reference.
  if (tuple)
  {
    for (Py_ssize_t i = 0; i < len; ++i)
    {
      PyObject *arg = pyObjectVector[i];
      Py_XINCREF(arg);
      PyTuple_SetItem(tuple, i, arg); // It steals a reference to arg
    }
  }
  else
  {
    PyErr_Print();
    UMUQFAILRETURNNULL("Couldn't create python tuple.");
  }

  // Build pyObjectPairStringVector dict
  // Return a new empty dictionary, or NULL on failure.
  PyObject *dict = PyDict_New();
  if (dict)
  {
    for (auto keyval : pyObjectPairStringVector)
    {
      PyDict_SetItemString(dict, keyval.first.c_str(), keyval.second);
    }
  }
  else
  {
    PyErr_Print();
    UMUQFAILRETURNNULL("Couldn't create python dictionary.");
  }

  // Call a callable Python object, with arguments given by the tuple args,
  // and named arguments given by the dictionary kwargs. Return the result
  // of the call on success, or raise an exception and return NULL on failure.
  PyObject *res = PyObject_Call(function, tuple, dict);
  if (!res)
  {
    if (PyErr_Occurred())
    {
      PyErr_Print();
      UMUQFAILRETURNNULL("Exception in calling a python function.");
    }
    UMUQFAILRETURNNULL("Failed to call the function.");
  }
  return res;
}

/*!
 * \brief Call a function using a function name
 *
 * \param functionName A callable function name
 * \param pyObjectVector Vector of variable-length argument list
 * \param pyObjectPairStringVector Vector of keyworded, variable-length argument list
 *
 * \return PyObject*
 */
PyObject *PyCallFunctionName(std::string const &functionName,
                             PyObjectVector const &pyObjectVector,
                             PyObjectPairStringVector const &pyObjectPairStringVector)
{
  // Check the name size
  if (!functionName.size())
  {
    return NULL;
  }

  PyObject *module = NULL;

  // Load the longest prefix of name that is a valid module name.
  std::string moduleName;

  auto nSize = functionName.size();
  while (!module && nSize != std::string::npos)
  {
    nSize = functionName.rfind('.', nSize - 1);
    moduleName = functionName.substr(0, nSize);
    module = PyImport_ImportModule(moduleName.c_str()); // Return value: New reference.
  }

  // A function object
  PyObject *function = NULL;

  if (module)
  {
    moduleName = functionName.substr(nSize);
    function = PyGetFunction(module, moduleName);
    if (!function)
    {
      UMUQFAILRETURNNULL("Importing the ", moduleName, " from ", functionName.substr(0, nSize), " failed.");
    }
  }
  else
  {
    PyObject *builtins = PyEval_GetBuiltins();                       // Return value: Borrowed reference.
    function = PyDict_GetItemString(builtins, functionName.c_str()); // Return value: Borrowed reference.
    Py_XINCREF(function);
    if (!function)
    {
      UMUQFAILRETURNNULL("Importing the ", functionName, " failed.");
    }
  }
  return PyCallFunctionObject(function, pyObjectVector, pyObjectPairStringVector);
}

template <typename... Args>
PyObject *PyCallFunctionName(std::string const &functionName, Args... args)
{
  PyObjectVector pyObjectVector;
  PyObjectPairStringVector pyObjectPairStringVector;
  appendArgs(pyObjectVector, pyObjectPairStringVector, args...);
  return PyCallFunctionName(functionName, pyObjectVector, pyObjectPairStringVector);
}

/*!
 * \brief Call a python function from module from
 *
 * \param functionName A callable function name
 * \param from Module to call the function from it
 * \param pyObjectVector Vector of variable-length argument list
 * \param pyObjectPairStringVector Vector of keyworded, variable-length argument list
 * \return PyObject*
 */
PyObject *PyCallFunctionNameFrom(std::string const &functionName,
                                 PyObject *from,
                                 PyObjectVector const &pyObjectVector,
                                 PyObjectPairStringVector const &pyObjectPairStringVector)
{
  std::string name = (functionName[0] == '.') ? functionName : '.' + functionName;

  PyObject *function = PyGetFunction(from, name);
  if (!function)
  {
    UMUQFAILRETURNNULL("Lookup of function ", functionName, " failed.");
  }
  return PyCallFunctionObject(function, pyObjectVector, pyObjectPairStringVector);
}

template <typename... Args>
PyObject *PyCallFunctionNameFrom(std::string const &functionName, PyObject *from, Args... args)
{
  PyObjectVector pyObjectVector;
  PyObjectPairStringVector pyObjectPairStringVector;
  appendArgs(pyObjectVector, pyObjectPairStringVector, args...);
  return PyCallFunctionNameFrom(functionName, from, pyObjectVector, pyObjectPairStringVector);
}

/*!
 * \brief Convert python tuple to the vector
 *
 * \param PyTuple Python tuple object
 * \return PyObjectVector
 */
PyObjectVector PyTupleToVector(PyObject *PyTuple)
{
  // Return true if p is a tuple object or an instance of a subtype of the tuple type.
  if (!PyTuple_Check(PyTuple))
  {
    UMUQFAIL("Input argument is not a python tuple.");
  }

  Py_ssize_t const tSize = PyTuple_Size(PyTuple);
  PyObjectVector Vector(tSize);
  for (Py_ssize_t i = 0; i < tSize; ++i)
  {
    Vector[i] = PyTuple_GetItem(PyTuple, i); // Return value: Borrowed reference.
  }
  return Vector;
}

/*!
 * \brief Convert python dictionary to the vector of pairs
 *
 * \param PyDict Python dictionary object
 * \return PyObjectMapChar
 */
PyObjectMapChar PyDictToMap(PyObject *PyDict)
{
  // Return true if p is a dict object or an instance of a subtype of the dict type.
  if (!PyDict_Check(PyDict))
  {
    UMUQFAIL("Input argument is not a python dict.");
  }

  PyObjectMapChar Map;
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(PyDict, &pos, &key, &value))
  {
    char const *str = PyString_AsString(key);
    PyObject *obj = value;
    Py_XINCREF(obj);
    Map.emplace(str, obj);
  }
  return Map;
}

/*!
 * \brief Convert a pointer, python tuple, and python dict to a pointer function
 *
 * \param Pointer A pointer to python object (a capsule)
 * \param PyTuple Python tuple object
 * \param PyDict Python dict object
 * \return PyObject*
 */
PyObject *PyPointerTupleDictToPointerFunctionVectorMap(PyObject *Pointer, PyObject *PyTuple, PyObject *PyDict)
{
  // Return true if its argument is a PyCapsule.
  if (!PyCapsule_CheckExact(Pointer))
  {
    UMUQFAIL("Pointer object is corrupted.");
  }

  // Retrieve the pointer stored in the capsule.
  PyObjectFunctionPointerP fun = reinterpret_cast<PyObjectFunctionPointerP>(PyCapsule_GetPointer(Pointer, NULL));
  // Return the current context stored in the capsule.
  void *PointerFunction = PyCapsule_GetContext(Pointer);
  auto Vec = PyTupleToVector(PyTuple);
  auto Map = PyDictToMap(PyDict);
  return fun(PointerFunction, Vec, Map);
}

/*!
 * \brief Convert a pointer, python tuple, and python dict to a pointer function
 *
 * \param Pointer A pointer to python object (a capsule)
 * \param PyTuple Python tuple object
 * \param PyDict Python dict object
 * \return PyObject*
 */
PyObject *PyPointerTupleDictToVectorMap(PyObject *Pointer, PyObject *PyTuple, PyObject *PyDict)
{
  // Return true if its argument is a PyCapsule.
  if (!PyCapsule_CheckExact(Pointer))
  {
    UMUQFAIL("Pointer object is corrupted.");
  }

  // Retrieve the pointer stored in the capsule.
  PyObjectFunctionPointer fun = reinterpret_cast<PyObjectFunctionPointer>(PyCapsule_GetPointer(Pointer, NULL));
  auto Vec = PyTupleToVector(PyTuple);
  auto Map = PyDictToMap(PyDict);
  return fun(Vec, Map);
}

// PyMethodDef: Structure used to describe a method of an extension type.
// PyCFunction: Type of the functions used to implement most Python callables in C.
// METH_KEYWORDS: Supports also keyword arguments.

PyMethodDef PyPointerTupleDictToPointerFunctionVectorMapMethod{
    "PFVecMap",                                                                  // name of the method
    reinterpret_cast<PyCFunction>(PyPointerTupleDictToPointerFunctionVectorMap), // pointer to the C implementation
    METH_KEYWORDS,                                                               // flag bits indicating how the call should be constructed
    nullptr                                                                      // points to the contents of the docstring
};

PyMethodDef PyPointerTupleDictToVectorMapMethod{
    "VecMap",                                                     // name of the method
    reinterpret_cast<PyCFunction>(PyPointerTupleDictToVectorMap), // pointer to the C implementation
    METH_KEYWORDS,                                                // flag bits indicating how the call should be constructed
    nullptr                                                       // points to the contents of the docstring
};

PyObject *PyLambda(PyObjectFunctionPointer fun)
{
  // Create a PyCapsule encapsulating the pointer. The pointer argument may not be NULL.
  PyObject *capsule = PyCapsule_New(reinterpret_cast<void *>(fun), NULL, NULL); // Return value: New reference.

  return PyCFunction_New(&PyPointerTupleDictToVectorMapMethod, capsule);
}

PyObject *PyLambdaP(PyObjectFunctionPointerP fun, void *pointer)
{
  // Create a PyCapsule encapsulating the pointer. The pointer argument may not be NULL.
  PyObject *capsule = PyCapsule_New(reinterpret_cast<void *>(fun), pointer ? reinterpret_cast<char *>(pointer) : NULL, NULL);

  return PyCFunction_New(&PyPointerTupleDictToPointerFunctionVectorMapMethod, capsule);
}

inline PyObject *PyObjectConstruct(PyObjectFunctionPointer fun)
{
  return PyLambda(fun);
}

inline PyObject *PyObjectConstruct(PyObjectFunctionPointerP fun, void *pointer)
{
  return PyLambdaP(fun, pointer);
}

template <typename... Args>
void appendArgs(PyObjectVector &pyObjectVector, PyObjectPairStringVector &pyObjectPairStringVector, Args... args);

// Base case
template <>
inline void appendArgs(PyObjectVector & /* pyObjectVector */, PyObjectPairStringVector & /* pyObjectPairStringVector */)
{
}

template <typename Head, typename... Tail>
void appendArgs(PyObjectVector &pyObjectVector, PyObjectPairStringVector &pyObjectPairStringVector, Head head, Tail... tail)
{
  PyObject *arg = PyObjectConstruct(head);
  pyObjectVector.push_back(arg);
  appendArgs(pyObjectVector, pyObjectPairStringVector, tail...);
}

// Keyword argument from string
template <typename Head, typename... Tail>
void appendArgs(PyObjectVector &pyObjectVector, PyObjectPairStringVector &pyObjectPairStringVector, std::pair<std::string, Head> head, Tail... tail)
{
  PyObject *value = PyObjectConstruct(head.second);
  pyObjectPairStringVector.emplace_back(head.first, value);
  appendArgs(pyObjectVector, pyObjectPairStringVector, tail...);
}

// Keyword argument from const char*
template <typename Head, typename... Tail>
void appendArgs(PyObjectVector &pyObjectVector, PyObjectPairStringVector &pyObjectPairStringVector, std::pair<char const *, Head> head, Tail... tail)
{
  PyObject *value = PyObjectConstruct(head.second);
  pyObjectPairStringVector.emplace_back(std::string(head.first), value);
  appendArgs(pyObjectVector, pyObjectPairStringVector, tail...);
}

} // namespace python

namespace python
{
/*! \namespace umuq::python::numpy
 * \ingroup Python_Module
 *
 * \brief It contains several functions to convert a data array idata to
 * Python numpy array
 *
 */
inline namespace numpy
{
/*!
 * \ingroup Python_Module
 *
 * \brief Converts a data array idata to Python array
 *
 * \tparam DataType Data type
 *
 * \param idata Input array of data
 *
 * \return PyObject* Python array
 */
template <typename DataType>
PyObject *PyArray(std::vector<DataType> const &idata)
{
  PyObject *pArray;
  npy_intp nsize = static_cast<npy_intp>(idata.size());
  npy_intp PyArrayDims[] = {nsize};
  if (NPYDatatype<DataType> == NPY_NOTYPE)
  {
    pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<double>);
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    std::copy(idata.begin(), idata.end(), static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray))));
  }
  else
  {
    pArray = PyArray_SimpleNewFromData(1, PyArrayDims, NPYDatatype<DataType>, (void *)(idata.data()));
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNewFromData(...)' function failed");
    }
  }
  return pArray;
}

template <typename TIn, typename TOut>
PyObject *PyArray(std::vector<TIn> const &idata)
{
  PyObject *pArray;
  npy_intp nsize = static_cast<npy_intp>(idata.size());
  npy_intp PyArrayDims[] = {nsize};
  if (NPYDatatype<TOut> != NPYDatatype<TIn>)
  {
    if (NPYDatatype<TOut> == NPY_NOTYPE)
    {
      pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<double>);
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
      }
      std::copy(idata.begin(), idata.end(), static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray))));
    }
    else
    {
      pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<TOut>);
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
      }
      std::copy(idata.begin(), idata.end(), static_cast<TOut *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray))));
    }
  }
  else
  {
    pArray = PyArray_SimpleNewFromData(1, PyArrayDims, NPYDatatype<TIn>, (void *)(idata.data()));
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNewFromData(...)' function failed");
    }
  }
  return pArray;
}

/*!
 * \ingroup Python_Module
 *
 * \brief Converts a data idata to Python array of size nSize
 *
 * \tparam DataType Data type
 *
 * \param idata  Input data
 * \param nSize  Size of the requested array
 *
 * \return PyObject* Python array
 */
template <typename DataType>
PyObject *PyArray(DataType const idata, int const nSize)
{
  PyObject *pArray;
  npy_intp nsize = static_cast<npy_intp>(nSize);
  npy_intp PyArrayDims[] = {nsize};
  if (NPYDatatype<DataType> == NPY_NOTYPE)
  {
    pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<double>);
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    double *vd = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
    double const iData = static_cast<double>(idata);
    std::fill(vd, vd + nsize, iData);
  }
  else
  {
    pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<DataType>);
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    DataType *vd = static_cast<DataType *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
    std::fill(vd, vd + nsize, idata);
  }
  return pArray;
}

template <>
PyObject *PyArray<char>(char const idata, int const nSize)
{
  PyObject *pArray;
  npy_intp nsize = static_cast<npy_intp>(nSize);
  npy_intp PyArrayDims[] = {nsize};
  pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<char>);
  if (!pArray)
  {
    UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
  }
  char *vd = static_cast<char *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
  std::fill(vd, vd + nsize, idata);
  return pArray;
}

template <typename TIn, typename TOut>
PyObject *PyArray(TIn const idata, int const nSize)
{
  PyObject *pArray;
  npy_intp nsize = static_cast<npy_intp>(nSize);
  npy_intp PyArrayDims[] = {nsize};
  if (NPYDatatype<TOut> == NPY_NOTYPE)
  {
    pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<double>);
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    double *vd = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
    double const iData = static_cast<double>(idata);
    std::fill(vd, vd + nsize, iData);
  }
  else
  {
    pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<TOut>);
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    TOut *vd = static_cast<TOut *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
    TOut const iData = static_cast<TOut>(idata);
    std::fill(vd, vd + nsize, iData);
  }
  return pArray;
}

/*!
 * \ingroup Python_Module
 *
 * \brief Converts a data array idata to Python array
 *
 * \tparam DataType Data type
 *
 * \param idata   Input array of data
 * \param nSize   Size of the array
 * \param Stride  Element stride (default is 1)
 *
 * \return PyObject* Python array
 */
template <typename DataType>
PyObject *PyArray(DataType const *idata, int const nSize, std::size_t const Stride = 1)
{
  PyObject *pArray;
  if (Stride != 1)
  {
    arrayWrapper<DataType> iArray(idata, nSize, Stride);
    npy_intp nsize = static_cast<npy_intp>(iArray.size());
    npy_intp PyArrayDims[] = {nsize};
    if (NPYDatatype<DataType> == NPY_NOTYPE)
    {
      pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<double>);
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
      }
      double *vd = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
      std::copy(iArray.begin(), iArray.end(), vd);
    }
    else
    {
      pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<DataType>);
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
      }
      DataType *vd = static_cast<DataType *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
      std::copy(iArray.begin(), iArray.end(), vd);
    }
  }
  else
  {
    npy_intp nsize = static_cast<npy_intp>(nSize);
    npy_intp PyArrayDims[] = {nsize};
    if (NPYDatatype<DataType> == NPY_NOTYPE)
    {
      pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<double>);
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
      }
      double *vd = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
      std::copy(idata, idata + nsize, vd);
    }
    else
    {
      pArray = PyArray_SimpleNewFromData(1, PyArrayDims, NPYDatatype<DataType>, static_cast<void *>(idata));
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNewFromData(...)' function failed");
      }
    }
  }
  return pArray;
}

template <typename TIn, typename TOut>
PyObject *PyArray(TIn const *idata, int const nSize, std::size_t const Stride = 1)
{
  PyObject *pArray;
  if (Stride != 1)
  {
    arrayWrapper<TIn> iArray(idata, nSize, Stride);
    npy_intp nsize = static_cast<npy_intp>(iArray.size());
    npy_intp PyArrayDims[] = {nsize};
    if (NPYDatatype<TOut> == NPY_NOTYPE)
    {
      pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<double>);
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
      }
      double *vd = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
      std::copy(iArray.begin(), iArray.end(), vd);
    }
    else
    {
      pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<TOut>);
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
      }
      TOut *vd = static_cast<TOut *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
      std::copy(iArray.begin(), iArray.end(), vd);
    }
  }
  else
  {
    npy_intp nsize = static_cast<npy_intp>(nSize);
    npy_intp PyArrayDims[] = {nsize};
    if (NPYDatatype<TOut> != NPYDatatype<TIn>)
    {
      if (NPYDatatype<TOut> == NPY_NOTYPE)
      {
        pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<double>);
        if (!pArray)
        {
          UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
        }
        double *vd = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
        std::copy(idata, idata + nsize, vd);
      }
      else
      {
        pArray = PyArray_SimpleNew(1, PyArrayDims, NPYDatatype<TOut>);
        if (!pArray)
        {
          UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
        }
        TOut *vd = static_cast<TOut *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
        std::copy(idata, idata + nsize, vd);
      }
    }
    else
    {
      pArray = PyArray_SimpleNewFromData(1, PyArrayDims, NPYDatatype<TIn>, static_cast<void *>(idata));
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNewFromData(...)' function failed");
      }
    }
  }
  return pArray;
}

/*!
 * \ingroup Python_Module
 *
 * \brief Converts a data array idata to the Python 2D array
 *
 * \tparam DataType Data type
 *
 * \param idata  Input array of data (with size of nDimX*nDimY)
 * \param nDimX  X size in the 2D array
 * \param nDimY  Y size in the 2D array
 *
 * \returns PyObject* Python 2D array
 */
template <typename DataType>
PyObject *Py2DArray(std::vector<DataType> const &idata, int const nDimX, int const nDimY)
{
  if (idata.size() != static_cast<decltype(idata.size())>(nDimX) * nDimY)
  {
    UMUQFAIL("Data size does not match with mesh numbers!");
  }

  PyObject *pArray;
  npy_intp PyArrayDims[] = {nDimY, nDimX};
  if (NPYDatatype<DataType> == NPY_NOTYPE)
  {
    pArray = PyArray_SimpleNew(2, PyArrayDims, NPYDatatype<double>);
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    double *vd = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
    std::copy(idata.begin(), idata.end(), vd);
  }
  else
  {
    pArray = PyArray_SimpleNewFromData(2, PyArrayDims, NPYDatatype<DataType>, (void *)(idata.data()));
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNewFromData(...)' function failed");
    }
  }
  return pArray;
}

template <typename TIn, typename TOut>
PyObject *Py2DArray(std::vector<TIn> const &idata, int const nDimX, int const nDimY)
{
  if (idata.size() != static_cast<decltype(idata.size())>(nDimX) * nDimY)
  {
    UMUQFAIL("Data size does not match with mesh numbers!");
  }

  PyObject *pArray;
  npy_intp PyArrayDims[] = {nDimY, nDimX};
  if (NPYDatatype<TOut> != NPYDatatype<TIn>)
  {
    if (NPYDatatype<TOut> == NPY_NOTYPE)
    {
      pArray = PyArray_SimpleNew(2, PyArrayDims, NPYDatatype<double>);
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
      }
      double *vd = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
      std::copy(idata.begin(), idata.end(), vd);
    }
    else
    {
      pArray = PyArray_SimpleNew(2, PyArrayDims, NPYDatatype<TOut>);
      if (!pArray)
      {
        UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
      }
      TOut *vd = static_cast<TOut *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
      std::copy(idata.begin(), idata.end(), vd);
    }
  }
  else
  {
    pArray = PyArray_SimpleNewFromData(2, PyArrayDims, NPYDatatype<TOut>, (void *)(idata.data()));
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNewFromData(...)' function failed");
    }
  }
  return pArray;
}

template <typename DataType>
PyObject *Py2DArray(DataType const *idata, int const nDimX, int const nDimY)
{
  PyObject *pArray;
  npy_intp PyArrayDims[] = {nDimY, nDimX};
  if (NPYDatatype<DataType> == NPY_NOTYPE)
  {
    pArray = PyArray_SimpleNew(2, PyArrayDims, NPYDatatype<double>);
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNew()' function failed");
    }
    double *vd = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(pArray)));
    std::copy(idata, idata + nDimX * nDimY, vd);
  }
  else
  {
    pArray = PyArray_SimpleNewFromData(2, PyArrayDims, NPYDatatype<DataType>, static_cast<void *>(idata));
    if (!pArray)
    {
      UMUQFAILRETURNNULL("couldn't create a NumPy array: the 'PyArray_SimpleNewFromData(...)' function failed");
    }
  }
  return pArray;
}

} // namespace numpy
} // namespace python
} // namespace umuq

#endif // HAVE_PYTHON
#endif // UMUQ_PYTHON_H
