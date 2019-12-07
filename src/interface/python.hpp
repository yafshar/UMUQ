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
#ifdef PY_SSIZE_T_CLEAN
#undef PY_SSIZE_T_CLEAN
#endif // PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>
#include <pythread.h>

#ifdef isalnum
#undef isalnum
#undef isalpha
#undef islower
#undef isspace
#undef isupper
#undef tolower
#undef toupper
#endif // isalnum

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
#ifdef PyMethod_Check
#undef PyMethod_Check
#endif // PyMethod_Check
#ifdef PyMethod_GET_FUNCTION
#undef PyMethod_GET_FUNCTION
#endif // PyMethod_GET_FUNCTION
#ifdef PyString_Check
#undef PyString_Check
#endif // PyString_Check
#ifdef PyString_FromStringAndSize
#undef PyString_FromStringAndSize
#endif // PyString_FromStringAndSize
#ifdef PyString_AsStringAndSize
#undef PyString_AsStringAndSize
#endif // PyString_AsStringAndSize
#ifdef PyString_Size
#undef PyString_Size
#endif // PyString_Size
#ifdef PyInt_Check
#undef PyInt_Check
#endif // PyInt_Check
#ifdef PyLong_AsLong
#undef PyLong_AsLong
#endif // PyLong_AsLong
#ifdef PyInt_FromSsize_t
#undef PyInt_FromSsize_t
#endif // PyInt_FromSsize_t
#ifdef PyInt_FromSize_t
#undef PyInt_FromSize_t
#endif // PyInt_FromSize_t
#ifdef PySliceObject
#undef PySliceObject
#endif // PySliceObject
#define PyString_FromString PyUnicode_FromString
#define PyString_AsString PyUnicode_AsUTF8
// #define PyString_AsString PyBytes_AsString
#define PyInt_FromLong PyLong_FromLong
#define PyInt_FromString PyLong_FromString
#define PyMethod_Check PyInstanceMethod_Check
#define PyMethod_GET_FUNCTION PyInstanceMethod_GET_FUNCTION
// #define PyMethod_New(ptr, nullptr, class_) PyInstanceMethod_New(ptr)
#define PyString_Check PyBytes_Check
#define PyString_FromStringAndSize PyBytes_FromStringAndSize
#define PyString_AsStringAndSize PyBytes_AsStringAndSize
#define PyString_Size PyBytes_Size
#define PyInt_Check PyLong_Check
#define PyLong_AsLong PyLong_AsLongLong
#define PyInt_FromSsize_t PyLong_FromSsize_t
#define PyInt_FromSize_t PyLong_FromSize_t
#define PySliceObject PyObject
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
 * \brief Type alias. A sorted container that contains key-value pairs of
 * 'char const *' and 'PyObject *', respectively
 *
 */
using PyObjectMapChar = std::map<char const *, PyObject *>;

/*!
 * \ingroup Python_Module
 *
 * \brief Type alias. A pair of 'string' and 'PyObject *' object
 *
 */
using PyObjectPairString = std::pair<std::string, PyObject *>;

/*!
 * \ingroup Python_Module
 *
 * \brief Type alias. A pair of 'char const *' and 'PyObject *' object
 *
 */
using PyObjectPairChar = std::pair<char const *, PyObject *>;

/*!
 * \ingroup Python_Module
 *
 * \brief Type alias. Vector of 'PyObject *' objects
 *
 */
using PyObjectVector = std::vector<PyObject *>;
static PyObjectVector const PyObjectVectorEmpty(PyObjectVector{});

/*!
 * \ingroup Python_Module
 *
 * \brief Type alias. Vector of 'PyObjectPairString' objects
 *
 */
using PyObjectPairStringVector = std::vector<PyObjectPairString>;
static PyObjectPairStringVector const PyObjectPairStringVectorEmpty(PyObjectPairStringVector{});

/*!
 * \ingroup Python_Module
 *
 * \brief Type alias. Vector of 'PyObjectPairChar' objects
 *
 */
using PyObjectPairCharVector = std::vector<PyObjectPairChar>;
static PyObjectPairCharVector const PyObjectPairCharVectorEmpty(PyObjectPairCharVector{});

/*!
 * \ingroup Python_Module
 *
 * \brief A function pointer to a function of 'PyObjectVector', and 'PyObjectMapChar' inputs.
 *
 */
using PyObjectFunctionPointer = PyObject *(*)(PyObjectVector const &pyObjectVector, PyObjectMapChar const &pyObjectMapChar);

/*!
 * \ingroup Python_Module
 *
 * \brief A function pointer to a function of 'void *', 'PyObjectVector', and 'PyObjectMapChar' inputs.
 *
 */
using PyObjectFunctionPointerP = PyObject *(*)(void *PointerFunction, PyObjectVector const &pyObjectVector, PyObjectMapChar const &pyObjectMapChar);

/*!
 * \brief Construct a Python object from an 'int' data value
 *
 * \param data Input data
 * \return PyObject*
 */
inline PyObject *PyObjectConstruct(int const data)
{
  return PyInt_FromLong(data);
}

/*!
 * \ingroup Python_Module
 *
 * \brief Construct a Python object from an 'size_t' data value
 *
 * \param data Input data
 * \return PyObject*
 */
inline PyObject *PyObjectConstruct(std::size_t const data)
{
  return PyLong_FromSize_t(data);
}

/*!
 * \ingroup Python_Module
 *
 * \brief Construct a Python object from a 'float' data value
 *
 * \param data Input data
 * \return PyObject*
 */
inline PyObject *PyObjectConstruct(float const data)
{
  return PyFloat_FromDouble(data);
}

/*!
 * \ingroup Python_Module
 *
 * \brief Construct a Python object from a 'double' data value
 *
 * \param data Input data
 * \return PyObject*
 */
inline PyObject *PyObjectConstruct(double const data)
{
  return PyFloat_FromDouble(data);
}

/*!
 * \ingroup Python_Module
 *
 * \brief Construct a Python object from a 'string' data
 *
 * \param data Input data
 * \return PyObject*
 */
inline PyObject *PyObjectConstruct(std::string const &data)
{
  return PyString_FromString(data.c_str());
}

/*!
 * \ingroup Python_Module
 *
 * \brief Construct a Python object from 'char const *' data
 *
 * \param data Input data
 * \return PyObject*
 */
inline PyObject *PyObjectConstruct(char const *data)
{
  return PyString_FromString(data);
}

/*!
 * \ingroup Python_Module
 *
 * \brief Construct a Python object from 'PyObjectVector' data
 *
 * \param data Input data
 * \return PyObject*
 */
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
    // Set the item at the index 'index' in the list to the new 'item'.
    // It “steals” a reference to item
    PyList_SetItem(list, index, item);
  }
  return list;
}

/*!
 * \ingroup Python_Module
 *
 * \brief Construct a Python object from 'PyObject' data
 *
 * \param data Input data
 * \return PyObject*
 */
inline PyObject *PyObjectConstruct(PyObject *data)
{
  return data;
}

/*!
 * \ingroup Python_Module
 *
 * \brief Get an attribute in module name 'moduleName' from a 'module' object
 *
 * Get an attribute in a module name from 'module' object. For example, to
 * get an attribute named 'echofilter' from 'module' object,
 * \code{.cpp}
 * std::string moduleName = "echofilter";
 * auto object = PyGetAttribute(module, moduleName);
 * \endcode
 *
 * Also, we can evaluate the chain of dot-operators that leads from the
 * module to the attribute. For example, to get an attribute named
 * 'echofilter' in a module name of 'sound.effects.echo.echofilter' from a
 * 'module' object,
 * \code{.cpp}
 * std::string moduleName = "sound.effects.echo.echofilter";
 * auto object = PyGetAttribute(module, moduleName);
 * \endcode
 *
 * \param module Object to retrive the attribute from it.
 * \param moduleName Module name.
 *
 * \return PyObject*
 *
 */
PyObject *PyGetAttribute(PyObject *module, std::string const &moduleName)
{
  // Check the name size
  if (!moduleName.size())
  {
    UMUQWARNING("The input moduleName is empty.");
    return NULL;
  }
  PyObject *object = module;
  std::size_t find_dot = 0;
  if (moduleName[0] != '.')
  {
    std::size_t const next_dot = moduleName.find('.', 0);
    std::string const attr_name = moduleName.substr(0, next_dot);
    // Retrieve an attribute named 'attr_name' from object 'object'.
    object = PyObject_GetAttrString(object, attr_name.c_str()); // Return value: New reference.
    find_dot = next_dot;
  }
  while (find_dot != std::string::npos)
  {
    std::size_t const next_dot = moduleName.find('.', find_dot + 1);
    std::string const attr_name = moduleName.substr(find_dot + 1, next_dot - (find_dot + 1));
    // Retrieve an attribute named 'attr_name' from object 'object'.
    object = PyObject_GetAttrString(object, attr_name.c_str()); // Return value: New reference.
    find_dot = next_dot;
  }
  return object;
}

/*!
 * \ingroup Python_Module
 *
 * \brief Call a callable python function
 *
 * \param function A callable function object
 * \param pyObjectVector Vector of variable-length argument list
 * \param pyObjectPairStringVector Vector of keyworded, variable-length argument list
 *
 * \return PyObject*
 *
 * \note
 * \c function should be a callable python function.
 * This function doesn't perform checks on the return value, only input is
 * checked.
 */
PyObject *PyCallFunctionObject(PyObject *function,
                               PyObjectVector const &pyObjectVector = PyObjectVectorEmpty,
                               PyObjectPairStringVector const &pyObjectPairStringVector = PyObjectPairStringVectorEmpty)
{
  // Determine if the function is callable.
  // Return 1 if the function is callable and 0 otherwise. (always succeeds.)
  if (!PyCallable_Check(function))
  {
    UMUQFAILRETURNNULL("The input function isn't callable.");
  }

  // If there is an extra argument or keywords
  Py_ssize_t const pyObjectVectorSize = static_cast<Py_ssize_t>(pyObjectVector.size());
  Py_ssize_t const pyObjectPairStringVectorSize = static_cast<Py_ssize_t>(pyObjectPairStringVector.size());

  if (!pyObjectVectorSize && !pyObjectPairStringVectorSize)
  {
    // Call a callable Python object, with args can be NULL.
    PyObject *res = PyObject_CallObject(function, NULL);
    if (!res)
    {
      if (PyErr_Occurred())
      {
        UMUQFAILRETURNNULL("Exception in calling a python function.");
      }
      UMUQFAILRETURNNULL("Failed to call the function.");
    }
    return res;
  }
  else
  {
    // Return a new tuple object of size 'pyObjectVectorSize',
    // or NULL on failure.
    PyObject *tuple = PyTuple_New(pyObjectVectorSize); // Return value: New reference.
    if (tuple)
    {
      for (Py_ssize_t pos = 0; pos < pyObjectVectorSize; ++pos)
      {
        PyObject *obj = pyObjectVector[pos];
        // Increment the reference count for object 'obj'
        Py_XINCREF(obj);
        // Insert a reference to object 'obj' at position 'pos' of the tuple
        // pointed to by 'tuple'.
        PyTuple_SetItem(tuple, pos, obj); // It steals a reference to arg
      }
    }
    else
    {
      UMUQFAILRETURNNULL("Couldn't create a python tuple.");
    }

    // Build pyObjectPairStringVector dict
    // Return a new empty dictionary, or NULL on failure.
    PyObject *dict = PyDict_New();
    if (dict)
    {
      for (auto keyval : pyObjectPairStringVector)
      {
        // Insert 'keyval.second' into the dictionary 'dict' using
        // 'keyval.first' as a key.
        PyDict_SetItemString(dict, keyval.first.c_str(), keyval.second);
      }
    }
    else
    {
      UMUQFAILRETURNNULL("Couldn't create a python dictionary.");
    }

    // Call a callable Python object, with arguments given by the tuple args,
    // and named arguments given by the dictionary kwargs. Return the result
    // of the call on success, or raise an exception and return NULL on failure.
    PyObject *res = PyObject_Call(function, tuple, dict);
    if (!res)
    {
      if (PyErr_Occurred())
      {
        UMUQFAILRETURNNULL("Exception in calling a python function.");
      }
      UMUQFAILRETURNNULL("Failed to call the function.");
    }
    return res;
  }
}

/*!
 * \ingroup Python_Module
 *
 * \brief Call a function using a function name
 *
 * Call a function using a function name, for example, to call a function
 * named 'echofilter' from the submodule 'sound.effects.echo', then the
 * function name input to this function is
 * \code{.cpp}
 * std::string functionName = "sound.effects.echo.echofilter";
 * \endcode
 *
 * \param functionName A callable function name with a full names of
 * individual submodules.
 * \param pyObjectVector Vector of variable-length argument list
 * \param pyObjectPairStringVector Vector of keyworded, variable-length argument list
 *
 * \return PyObject*
 *
 */
PyObject *PyCallFunctionName(std::string const &functionName,
                             PyObjectVector const &pyObjectVector = PyObjectVectorEmpty,
                             PyObjectPairStringVector const &pyObjectPairStringVector = PyObjectPairStringVectorEmpty)
{
  // Check the name size
  if (!functionName.size())
  {
    UMUQWARNING("The input functionName is empty.");
    return NULL;
  }

  PyObject *module = NULL;

  auto nSize = functionName.size();
  while (!module && nSize != std::string::npos)
  {
    nSize = functionName.rfind('.', nSize - 1);

    std::string const submodule = functionName.substr(0, nSize);

    // The return value is a new reference to the imported module
    module = PyImport_ImportModule(submodule.c_str()); // Return value: New reference.
  }

  // A function object
  PyObject *function = NULL;

  if (module)
  {
    // Load the longest prefix of name that is a valid module name.
    std::string moduleName = functionName.substr(nSize);

    function = PyGetAttribute(module, moduleName);
    if (!function)
    {
      UMUQFAILRETURNNULL("Importing the '", moduleName, "' from '", functionName.substr(0, nSize), "' submodules failed.");
    }
  }
  else
  {
    // Return a dictionary of the builtins in the current execution frame
    PyObject *builtins = PyEval_GetBuiltins(); // Return value: Borrowed reference.

    // Return the object 'function' from the dictionary 'builtins' which has
    // a key 'functionName'.
    function = PyDict_GetItemString(builtins, functionName.c_str()); // Return value: Borrowed reference.

    // Increment the reference count for object 'function'
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
 * \ingroup Python_Module
 *
 * \brief Call a python function from module 'module'
 *
 * \param functionName A callable function name
 * \param module Module to call the function from it
 * \param pyObjectVector Vector of variable-length argument list
 * \param pyObjectPairStringVector Vector of keyworded, variable-length argument list
 * \return PyObject*
 */
PyObject *PyCallFunctionNameFromModule(std::string const &functionName,
                                       PyObject *module,
                                       PyObjectVector const &pyObjectVector = PyObjectVectorEmpty,
                                       PyObjectPairStringVector const &pyObjectPairStringVector = PyObjectPairStringVectorEmpty)
{
  PyObject *function = PyGetAttribute(module, functionName);
  if (!function)
  {
    UMUQFAILRETURNNULL("Failed to retrieve an attribute named '", functionName, "' from the 'module' object.");
  }
  return PyCallFunctionObject(function, pyObjectVector, pyObjectPairStringVector);
}

template <typename... Args>
PyObject *PyCallFunctionNameFromModule(std::string const &functionName, PyObject *module, Args... args)
{
  PyObjectVector pyObjectVector;
  PyObjectPairStringVector pyObjectPairStringVector;
  appendArgs(pyObjectVector, pyObjectPairStringVector, args...);
  return PyCallFunctionNameFromModule(functionName, module, pyObjectVector, pyObjectPairStringVector);
}

/*!
 * \ingroup Python_Module
 *
 * \brief Call a python function from module name 'moduleName'
 *
 * \param functionName A callable function name
 * \param moduleName Module name to call the function from it
 * \param pyObjectVector Vector of variable-length argument list
 * \param pyObjectPairStringVector Vector of keyworded, variable-length argument list
 * \return PyObject*
 */
PyObject *PyCallFunctionNameFromModuleName(std::string const &functionName,
                                           std::string const &moduleName,
                                           PyObjectVector const &pyObjectVector = PyObjectVectorEmpty,
                                           PyObjectPairStringVector const &pyObjectPairStringVector = PyObjectPairStringVectorEmpty)
{
  // Check the name size
  if (!functionName.size())
  {
    UMUQWARNING("The input functionName is empty.");
    return NULL;
  }
  if (!moduleName.size())
  {
    UMUQWARNING("The input moduleName is empty.");
    return NULL;
  }

  // The return value is a new reference to the imported module
  PyObject *module = PyImport_ImportModule(moduleName.c_str()); // Return value: New reference.
  if (!module)
  {
    UMUQFAILRETURNNULL("Failed to import the '", moduleName, "' module.");
  }

  PyObject *function = PyGetAttribute(module, functionName);
  if (!function)
  {
    UMUQFAILRETURNNULL("Lookup of function '", functionName, "' failed.");
  }

  return PyCallFunctionObject(function, pyObjectVector, pyObjectPairStringVector);
}

template <typename... Args>
PyObject *PyCallFunctionNameFromModuleName(std::string const &functionName, std::string const &moduleName, Args... args)
{
  PyObjectVector pyObjectVector;
  PyObjectPairStringVector pyObjectPairStringVector;
  appendArgs(pyObjectVector, pyObjectPairStringVector, args...);
  return PyCallFunctionNameFromModuleName(functionName, moduleName, pyObjectVector, pyObjectPairStringVector);
}

/*!
 * \ingroup Python_Module
 *
 * \brief Convert python tuple to the vector of python objects
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
 * \ingroup Python_Module
 *
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
    // Increment the reference count for object 'function'
    Py_XINCREF(obj);
    Map.emplace(str, obj);
  }
  return Map;
}

/*!
 * \ingroup Python_Module
 *
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
 * \ingroup Python_Module
 *
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

/*!
 * \ingroup Python_Module
 *
 * \brief Construct a Python object from 'PyObjectFunctionPointer' data
 *
 * \param data Input data
 * \return PyObject*
 */
inline PyObject *PyObjectConstruct(PyObjectFunctionPointer fun)
{
  return PyLambda(fun);
}

/*!
 * \ingroup Python_Module
 *
 * \brief Construct a Python object from 'PyObjectFunctionPointerP' data
 *
 * \param data Input data
 * \return PyObject*
 */
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
