#ifndef UMUQ_MATPLOTLIB_H
#define UMUQ_MATPLOTLIB_H
#ifdef HAVE_PYTHON
/*!
 * \file io/matplotlib.hpp
 * \brief 
 * 
 * 
 * The matplotlib Module contains the modification to the original matplotlib source 
 * codes made available under the following license:
 * 
 * \verbatim
 * The MIT License (MIT)
 * 
 * Copyright (c) 2014 Benno Evers
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * \endverbatim
 */

#include "../core/core.hpp"
#include "../misc/array.hpp"

/*!
 * \brief Type selector for numpy array conversion
 * 
 * \tparam T Data type
 */
template <typename T>
constexpr NPY_TYPES NPIDatatype = NPY_NOTYPE; // variable template

template <>
constexpr NPY_TYPES NPIDatatype<bool> = NPY_BOOL;

template <>
constexpr NPY_TYPES NPIDatatype<int8_t> = NPY_INT8;

template <>
constexpr NPY_TYPES NPIDatatype<int16_t> = NPY_SHORT;

template <>
constexpr NPY_TYPES NPIDatatype<int32_t> = NPY_INT;

template <>
constexpr NPY_TYPES NPIDatatype<int64_t> = NPY_INT64;

template <>
constexpr NPY_TYPES NPIDatatype<uint8_t> = NPY_UINT8;

template <>
constexpr NPY_TYPES NPIDatatype<uint16_t> = NPY_USHORT;

template <>
constexpr NPY_TYPES NPIDatatype<uint32_t> = NPY_ULONG;

template <>
constexpr NPY_TYPES NPIDatatype<uint64_t> = NPY_UINT64;

template <>
constexpr NPY_TYPES NPIDatatype<float> = NPY_FLOAT;

template <>
constexpr NPY_TYPES NPIDatatype<double> = NPY_DOUBLE;

/*!
 * \brief Converts a data array idata to Python array
 * 
 * \tparam T Data type
 * 
 * \param idata Input array of data
 * 
 * \return PyObject* 
 */
template <typename T>
PyObject *PyArray(std::vector<T> const &idata)
{
	PyObject *pArray;
	{
		npy_intp nsize = static_cast<npy_intp>(idata.size());
		if (NPIDatatype<T> == NPY_NOTYPE)
		{
			std::vector<double> vd(nsize);
			std::copy(idata.begin(), idata.end(), vd.begin());
			pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_DOUBLE, (void *)(vd.data()));
		}
		else
		{
			pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<T>, (void *)(idata.data()));
		}
	}
	return pArray;
}

/*!
 * \brief Converts a data array idata to Python array
 * 
 * \tparam T Data type
 * 
 * \param idata array of data
 * \param nSize size of the array
 * \param Stride element stride (default is 1)
 * 
 * \return PyObject* 
 */
template <typename T>
PyObject *PyArray(T *idata, int const nSize, std::size_t const Stride = 1)
{
	PyObject *pArray;
	{
		npy_intp nsize;

		if (Stride != 1)
		{
			ArrayWrapper<T> iArray(idata, nSize, Stride);
			nsize = static_cast<npy_intp>(iArray.size());
			if (NPIDatatype<T> == NPY_NOTYPE)
			{
				std::vector<double> vd(nsize);
				std::copy(iArray.begin(), iArray.end(), vd.begin());
				pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_DOUBLE, (void *)(vd.data()));
			}
			else
			{
				std::vector<T> vd(nsize);
				std::copy(iArray.begin(), iArray.end(), vd.begin());
				pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<T>, (void *)(vd.data()));
			}
			return pArray;
		}

		nsize = static_cast<npy_intp>(nSize);
		if (NPIDatatype<T> == NPY_NOTYPE)
		{
			std::vector<double> vd(nsize);
			std::copy(idata, idata + nSize, vd.begin());
			pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_DOUBLE, (void *)(vd.data()));
		}
		else
		{
			pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<T>, (void *)(idata));
		}
	}
	return pArray;
}

/*! \class pyplot
 * \brief This module contains functions that allow you to generate many kinds of plots quickly.
 * 
 * 
 * Documents Reference:
 * https://matplotlib.org/api/pyplot_summary.html
 */
class pyplot
{
  public:
	/*!
     * \brief Construct a new pyplot object
     * 
     */
	pyplot() {}

	/*!
	 * \brief Destroy the pyplot object
	 * 
	 */
	~pyplot() {}

	/*!
     * \brief Annotate the point xy with text s
     * 
     * \tparam Data type
     * 
     * \param annotation  The text of the annotation
     * \param x           x point (x,y) to annotate
     * \param y           y point (x,y) to annotate
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool annotate(std::string const &annotation, T x, T y);

	/*!
     * \brief Convenience method to get or set axis properties
     * 
     * \param axisArguments 
     */
	inline bool axis(std::string const &axisArguments);

	/*!
     * \brief Clear the current figure
     * 
     */
	inline bool clf();

	/*!
     * \brief Close a figure window
     * 
     */
	inline bool close();

	/*!
     * \brief Redraw the current figure
     * This is used to update a figure that has been altered, 
     * but not automatically re-drawn. If interactive mode is 
     * on (ion()), this should be only rarely needed
     * 
     */
	inline bool draw();

	/*!
     * \brief Plot y versus x as lines and/or markers with attached errorbars
     * 
     * \tparam T      Data type
     * 
     * \param x       Scalar or array-like, data positions
     * \param y       Scalar or array-like, data positions
     * \param yerr    Errorbar
     * \param fmt     Plot format string
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool errorbar(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &yerr, std::string const &fmt = "");

	/*!
     * \brief Plot y versus x as lines and/or markers with attached errorbars
     * 
     * \tparam T      Data type
     * 
     * \param x       Scalar or array-like, data positions
     * \param y       Scalar or array-like, data positions
     * \param yerr    Errorbar
     * \param fmt     Plot format string
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool errorbar(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &yerr, std::string const &fmt = "");

	/*!
     * \brief Creates a new figure
     * 
     */
	inline bool figure();

	/*!
	 * \brief Creates a new figure
	 * 
	 * \param width   width in inches
	 * \param height  height in inches
	 * \param dpi     resolution of the figure (default is 100)
	 * 
	 * \return true 
	 * \return false 
	 */
	bool figure(std::size_t const width, std::size_t const height, std::size_t const dpi = 100);

	/*!
     * \brief Fill the area between two horizontal curves.
     * The curves are defined by the points (x, y1) and (x, y2). 
     * This creates one or multiple polygons describing the filled area.
     * 
     * \tparam T 
     * 
     * \param x          The x coordinates of the nodes defining the curves.
     * \param y1         The y coordinates of the nodes defining the first curve.
     * \param y2         The y coordinates of the nodes defining the second curve.
     * \param keywords   All other keyword arguments are passed on to PolyCollection. They control the Polygon properties
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool fill_between(std::vector<T> const &x, std::vector<T> const &y1, std::vector<T> const &y2, std::map<std::string, std::string> const &keywords);

	/*!
     * \brief Turn the axes grids on or off
     * 
     * \param flag 
     */
	bool grid(bool flag);

	/*!
     * \brief Plot a histogram
     * Compute and draw the histogram of x. 
     * 
     * \tparam T 
     * 
     * \param x       Input values
     * \param bins    The bin specification (The default value is 10)
     * \param color   Color or array_like of colors or None, optional
	 * \param label   default is None
     * \param alpha   The alpha blending value \f$ 0 <= scalar <= 1 \f$ or None, optional
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool hist(std::vector<T> const &x, long bins = 10, std::string const &color = "b", std::string const &label = "", double const alpha = 1.0);

	/*!
     * \brief Turn interactive mode on
     * 
     */
	inline bool ion();

	/*!
     * \brief Places a legend on the axes
     * 
     */
	inline bool legend();

	/*!
     * \brief Make a plot with log scaling on both the x and y axis
     * This is just a thin wrapper around plot which additionally changes 
     * both the x-axis and the y-axis to log scaling. All of the concepts 
     * and parameters of plot can be used here as well.
     * 
     * \tparam T
     * 
     * \param x 
     * \param y 
     * \param fmt
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool loglog(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt = "");

	/*!
     * \brief Pause for interval seconds.
     * 
     * \tparam T 
     * 
     * \param interval 
     */
	inline bool pause(double const interval);

	/*!
     * \brief Plot y versus x as lines and/or markers.
     * 
     * \tparam T 
     * 
     * \param x         array-like or scalar
     * \param y         array-like or scalar
     * \param keywords  are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool plot(std::vector<T> const &x, std::vector<T> const &y, std::map<std::string, std::string> const &keywords);

	/*!
     * \brief Plot y versus x as lines and/or markers.
     * 
     * \tparam T 
     * 
     * \param x         array-like or scalar
     * \param y         array-like or scalar
     * \param fmt       A format string, e.g. ‘ro’ for red circles.
     *                  Format strings are just an abbreviation for quickly setting basic line properties. 
     * \param label     object. Set the label to s for auto legend
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool plot(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt = "", std::string const &label = "");

	/*!
     * \brief Save the current figure
     * 
     * \param filename A string containing a path to a filename
     */
	bool savefig(std::string const &filename);

	/*!
     * \brief Make a plot with log scaling on the x axis
     * This is just a thin wrapper around plot which additionally changes the x-axis to log scaling. 
     * All of the concepts and parameters of plot can be used here as well.
     * 
     * \tparam T
     * 
     * \param x 
     * \param y 
     * \param fmt  
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool semilogx(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt = "");

	/*!
     * \brief Make a plot with log scaling on the y axis
     * This is just a thin wrapper around plot which additionally changes the x-axis to log scaling. 
     * All of the concepts and parameters of plot can be used here as well.
     * 
     * \tparam T
     * 
     * \param x 
     * \param y 
     * \param fmt  
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool semilogy(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt = "");

	/*!
     * \brief Display a figure. 
     * 
     * \param block 
     */
	bool show(bool const block = true);

	/*!
     * \brief Create a stem plot.
     * A stem plot plots vertical lines at each x location from the baseline to y, and places a marker there
     * 
     * \tparam T 
     * 
     * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1)
     * \param y         The y-values of the stem heads
     * \param keywords  
     * 
     * \return true 
     * \return false 
     */
	template <typename T>
	bool stem(std::vector<T> const &x, std::vector<T> const &y, std::map<std::string, std::string> const &keywords);

	template <typename T>
	bool stem(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt = "");

	/*!
     * \brief Return a subplot axes at the given grid position
     * In the current figure, create and return an Axes, at position index of a (virtual) grid of nrows by ncols axes. 
     * Indexes go from 1 to nrows * ncols, incrementing in row-major order.
     * 
     * \param nrows 
     * \param ncols 
     * \param index 
     */
	bool subplot(long nrows, long ncols, long index);

	/*!
     * \brief Set a title of the current axes
     * 
     * \param label Text to use for the title
     */
	bool title(std::string const &label);

	// Actually, is there any reason not to call this automatically for every plot?
	/*!
     * \brief Automatically adjust subplot parameters to give specified padding.
     * 
     */
	inline bool tight_layout();

	/*!
     * \brief Set the x limits of the current axes
     * 
     * \tparam T Data type
     * 
     * \param left  xmin
     * \param right xmax
     */
	template <typename T>
	bool xlim(T left, T right);

	/*!
     * \brief Get the x limits of the current axes
     * 
     * \tparam T Data type
     * 
     * \param left  xmin
     * \param right xmax
     */
	template <typename T>
	bool xlim(T *left, T *right);

	/*!
     * \brief Set the x-axis label of the current axes
     * 
     * \param label The label text
     */
	bool xlabel(std::string const &label);

	/*!
     * \brief Turns on xkcd sketch-style drawing mode
     * This will only have effect on things drawn after this function is called
     * 
     */
	inline bool xkcd();

	/*!
     * \brief Set the y limits of the current axes
     * 
     * \tparam T Data type
     * 
     * \param left  ymin
     * \param right ymax
     */
	template <typename T>
	bool ylim(T left, T right);

	/*!
     * \brief Get the y limits of the current axes
     * 
     * \tparam T Data type
     * 
     * \param left  ymin
     * \param right ymax
     */
	template <typename T>
	bool ylim(T *left, T *right);

	/*!
     * \brief Set the y-axis label of the current axes
     * 
     * \param label The label text
     */
	bool ylabel(std::string const &label);

  public:
	//! An instance of matplotlib_interpreter object
	static matplotlib_interpreter mpl;

  private:
	class matplotlib_interpreter
	{
	  public:
		matplotlib_interpreter()
		{
			// optional but recommended
#if PY_MAJOR_VERSION >= 3
			wchar_t name[] = L"umuq";
#else
			char name[] = "umuq";
#endif

			//Pass name to the Python interpreter
			Py_SetProgramName(name);

			//Initialize the Python interpreter.  Required.
			Py_Initialize();

			//Initialize numpy
			import_array();

			PyObject *matplotlib = NULL;
			PyObject *pymod = NULL;
			PyObject *pylabmod = NULL;

			{
				PyObject *matplotlibname = PyString_FromString("matplotlib");
				if (!matplotlibname)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Couldnt create matplotlibname!" << std::endl;
					throw std::runtime_error("Couldnt create string!");
				}

				matplotlib = PyImport_Import(matplotlibname);
				if (!matplotlib)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Couldnt load module matplotlib!" << std::endl;
					throw std::runtime_error("Error loading module matplotlib!");
				}

				//Decrementing of the reference count
				Py_DECREF(matplotlibname);
			}

			{
				PyObject *pyplotname = PyString_FromString("matplotlib.pyplot");
				if (!pyplotname)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Couldnt create pyplotname!" << std::endl;
					throw std::runtime_error("Couldnt create string!");
				}

				pymod = PyImport_Import(pyplotname);
				if (!pymod)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Couldnt load module matplotlib.pyplot!" << std::endl;
					throw std::runtime_error("Error loading module matplotlib.pyplot!");
				}

				//Decrementing of the reference count
				Py_DECREF(pyplotname);
			}

			// matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
			// or matplotlib.backends is imported for the first time
			if (!s_backend.empty())
			{
				//Call the method named use of object matplotlib with a variable number of C arguments.
				PyObject_CallMethod(matplotlib, const_cast<char *>("use"), const_cast<char *>("s"), s_backend.c_str());
			}

			{
				PyObject *pylabname = PyString_FromString("pylab");
				if (!pylabname)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Couldnt create pylabname!" << std::endl;
					throw std::runtime_error("Couldnt create string!");
				}

				pylabmod = PyImport_Import(pylabname);
				if (!pylabmod)
				{
					std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
					std::cerr << " Couldnt load module pylab!" << std::endl;
					throw std::runtime_error("Error loading module pylab!");
				}

				//Decrementing of the reference count
				Py_DECREF(pylabname);
			}

			//Retrieve an attribute named show from object pymod.
			matplotlibFunction_show = PyObject_GetAttrString(pymod, "show");
			if (!matplotlibFunction_show)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find show function!" << std::endl;
				throw std::runtime_error("Couldn't find show function!");
			}
			//Return true if it is a function object
			if (!PyFunction_Check(matplotlibFunction_show))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named close from object pymod.
			matplotlibFunction_close = PyObject_GetAttrString(pymod, "close");
			if (!matplotlibFunction_close)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find close function!" << std::endl;
				throw std::runtime_error("Couldn't find close function!");
			}
			if (!PyFunction_Check(matplotlibFunction_close))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named draw from object pymod.
			matplotlibFunction_draw = PyObject_GetAttrString(pymod, "draw");
			if (!matplotlibFunction_draw)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find draw function!" << std::endl;
				throw std::runtime_error("Couldn't find draw function!");
			}
			if (!PyFunction_Check(matplotlibFunction_draw))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named pause from object pymod.
			matplotlibFunction_pause = PyObject_GetAttrString(pymod, "pause");
			if (!matplotlibFunction_pause)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find pause function!" << std::endl;
				throw std::runtime_error("Couldn't find pause function!");
			}
			if (!PyFunction_Check(matplotlibFunction_pause))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named figure from object pymod.
			matplotlibFunction_figure = PyObject_GetAttrString(pymod, "figure");
			if (!matplotlibFunction_figure)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find figure function!" << std::endl;
				throw std::runtime_error("Couldn't find figure function!");
			}
			if (!PyFunction_Check(matplotlibFunction_figure))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named plot from object pymod.
			matplotlibFunction_plot = PyObject_GetAttrString(pymod, "plot");
			if (!matplotlibFunction_plot)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find plot function!" << std::endl;
				throw std::runtime_error("Couldn't find plot function!");
			}
			if (!PyFunction_Check(matplotlibFunction_plot))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named semilogx from object pymod.
			matplotlibFunction_semilogx = PyObject_GetAttrString(pymod, "semilogx");
			if (!matplotlibFunction_semilogx)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find semilogx function!" << std::endl;
				throw std::runtime_error("Couldn't find semilogx function!");
			}
			if (!PyFunction_Check(matplotlibFunction_semilogx))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named semilogy from object pymod.
			matplotlibFunction_semilogy = PyObject_GetAttrString(pymod, "semilogy");
			if (!matplotlibFunction_semilogy)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find semilogy function!" << std::endl;
				throw std::runtime_error("Couldn't find semilogy function!");
			}
			if (!PyFunction_Check(matplotlibFunction_semilogy))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named loglog from object pymod.
			matplotlibFunction_loglog = PyObject_GetAttrString(pymod, "loglog");
			if (!matplotlibFunction_loglog)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find loglog function!" << std::endl;
				throw std::runtime_error("Couldn't find loglog function!");
			}
			if (!PyFunction_Check(matplotlibFunction_loglog))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named fill_between from object pymod.
			matplotlibFunction_fill_between = PyObject_GetAttrString(pymod, "fill_between");
			if (!matplotlibFunction_fill_between)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find fill_between function!" << std::endl;
				throw std::runtime_error("Couldn't find fill_between function!");
			}
			if (!PyFunction_Check(matplotlibFunction_fill_between))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named hist from object pymod.
			matplotlibFunction_hist = PyObject_GetAttrString(pymod, "hist");
			if (!matplotlibFunction_hist)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find hist function!" << std::endl;
				throw std::runtime_error("Couldn't find hist function!");
			}
			if (!PyFunction_Check(matplotlibFunction_hist))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named subplot from object pymod.
			matplotlibFunction_subplot = PyObject_GetAttrString(pymod, "subplot");
			if (!matplotlibFunction_subplot)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find subplot function!" << std::endl;
				throw std::runtime_error("Couldn't find subplot function!");
			}
			if (!PyFunction_Check(matplotlibFunction_subplot))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named legend from object pymod.
			matplotlibFunction_legend = PyObject_GetAttrString(pymod, "legend");
			if (!matplotlibFunction_legend)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find legend function!" << std::endl;
				throw std::runtime_error("Couldn't find legend function!");
			}
			if (!PyFunction_Check(matplotlibFunction_legend))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named ylim from object pymod.
			matplotlibFunction_ylim = PyObject_GetAttrString(pymod, "ylim");
			if (!matplotlibFunction_ylim)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find ylim function!" << std::endl;
				throw std::runtime_error("Couldn't find ylim function!");
			}
			if (!PyFunction_Check(matplotlibFunction_ylim))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named title from object pymod.
			matplotlibFunction_title = PyObject_GetAttrString(pymod, "title");
			if (!matplotlibFunction_title)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find title function!" << std::endl;
				throw std::runtime_error("Couldn't find title function!");
			}
			if (!PyFunction_Check(matplotlibFunction_title))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named axis from object pymod.
			matplotlibFunction_axis = PyObject_GetAttrString(pymod, "axis");
			if (!matplotlibFunction_axis)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find axis function!" << std::endl;
				throw std::runtime_error("Couldn't find axis function!");
			}
			if (!PyFunction_Check(matplotlibFunction_axis))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named xlabel from object pymod.
			matplotlibFunction_xlabel = PyObject_GetAttrString(pymod, "xlabel");
			if (!matplotlibFunction_xlabel)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find xlabel function!" << std::endl;
				throw std::runtime_error("Couldn't find xlabel function!");
			}
			if (!PyFunction_Check(matplotlibFunction_xlabel))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named ylabel from object pymod.
			matplotlibFunction_ylabel = PyObject_GetAttrString(pymod, "ylabel");
			if (!matplotlibFunction_ylabel)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find ylabel function!" << std::endl;
				throw std::runtime_error("Couldn't find ylabel function!");
			}
			if (!PyFunction_Check(matplotlibFunction_ylabel))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named grid from object pymod.
			matplotlibFunction_grid = PyObject_GetAttrString(pymod, "grid");
			if (!matplotlibFunction_grid)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find grid function!" << std::endl;
				throw std::runtime_error("Couldn't find grid function!");
			}
			if (!PyFunction_Check(matplotlibFunction_grid))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named xlim from object pymod.
			matplotlibFunction_xlim = PyObject_GetAttrString(pymod, "xlim");
			if (!matplotlibFunction_xlim)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find xlim function!" << std::endl;
				throw std::runtime_error("Couldn't find xlim function!");
			}
			if (!PyFunction_Check(matplotlibFunction_xlim))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named ion from object pymod.
			matplotlibFunction_ion = PyObject_GetAttrString(pymod, "ion");
			if (!matplotlibFunction_ion)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find ion function!" << std::endl;
				throw std::runtime_error("Couldn't find ion function!");
			}
			if (!PyFunction_Check(matplotlibFunction_ion))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named savefig from object pylabmod.
			matplotlibFunction_savefig = PyObject_GetAttrString(pylabmod, "savefig");
			if (!matplotlibFunction_savefig)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find savefig function!" << std::endl;
				throw std::runtime_error("Couldn't find savefig function!");
			}
			if (!PyFunction_Check(matplotlibFunction_savefig))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named annotate from object pymod.
			matplotlibFunction_annotate = PyObject_GetAttrString(pymod, "annotate");
			if (!matplotlibFunction_annotate)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find annotate function!" << std::endl;
				throw std::runtime_error("Couldn't find annotate function!");
			}
			if (!PyFunction_Check(matplotlibFunction_annotate))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named clf from object pymod.
			matplotlibFunction_clf = PyObject_GetAttrString(pymod, "clf");
			if (!matplotlibFunction_clf)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find clf function!" << std::endl;
				throw std::runtime_error("Couldn't find clf function!");
			}
			if (!PyFunction_Check(matplotlibFunction_clf))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named errorbar from object pymod.
			matplotlibFunction_errorbar = PyObject_GetAttrString(pymod, "errorbar");
			if (!matplotlibFunction_errorbar)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find errorbar function!" << std::endl;
				throw std::runtime_error("Couldn't find errorbar function!");
			}
			if (!PyFunction_Check(matplotlibFunction_errorbar))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named tight_layout from object pymod.
			matplotlibFunction_tight_layout = PyObject_GetAttrString(pymod, "tight_layout");
			if (!matplotlibFunction_tight_layout)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find tight_layout function!" << std::endl;
				throw std::runtime_error("Couldn't find tight_layout function!");
			}
			if (!PyFunction_Check(matplotlibFunction_tight_layout))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named stem from object pymod.
			matplotlibFunction_stem = PyObject_GetAttrString(pymod, "stem");
			if (!matplotlibFunction_stem)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find stem function!" << std::endl;
				throw std::runtime_error("Couldn't find stem function!");
			}
			if (!PyFunction_Check(matplotlibFunction_stem))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}
			//Retrieve an attribute named xkcd from object pymod.
			matplotlibFunction_xkcd = PyObject_GetAttrString(pymod, "xkcd");
			if (!matplotlibFunction_xkcd)
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Couldn't find xkcd function!" << std::endl;
				throw std::runtime_error("Couldn't find xkcd function!");
			}
			if (!PyFunction_Check(matplotlibFunction_xkcd))
			{
				std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
				std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
				throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
			}

			//Return a new tuple object of size 0
			pyEmptyTuple = PyTuple_New(0);
		}

		~matplotlib_interpreter()
		{
			//Undo all initializations made by Py_Initialize() and subsequent use
			//of Python/C API functions, and destroy all sub-interpreters
			Py_Finalize();
		}

		//! Must be called before the first regular call to matplotlib to have any effect
		/*!
		 * \brief Set the “backend” to any of user interface backends or hardcopy backends
		 * 
		 * \param name user interface backends (for use in pygtk, wxpython, tkinter, qt4, or macosx; 
		 *             also referred to as “interactive backends”) or hardcopy backends to make image 
		 *             files (PNG, SVG, PDF, PS; also referred to as “non-interactive backends”)
		 * 
         * Reference:
         * https://matplotlib.org/tutorials/introductory/usage.html
		 */
		inline void setbackend(std::string const &name)
		{
			backend = name;
		}

	  public:
		// Make it noncopyable
		matplotlib_interpreter(matplotlib_interpreter const &) = delete;

		// Make it not assignable
		matplotlib_interpreter &operator=(matplotlib_interpreter const &) = delete;
		void operator=(matplotlib_interpreter const &) = delete;

	  public:
		/*!
	     * To support all of use cases, matplotlib can target different outputs, and each of these capabilities is called a backend;
	     * the “frontend” is the user facing code, i.e., the plotting code, whereas the “backend” does all the hard work
	     * behind-the-scenes to make the figure.
	     * There are two types of backends: user interface backends (for use in pygtk, wxpython, tkinter, qt4, or macosx; 
	     * also referred to as “interactive backends”) and hardcopy backends to make image files (PNG, SVG, PDF, PS; also 
	     * referred to as “non-interactive backends”).
		 * 
         * Reference:
         * https://matplotlib.org/tutorials/introductory/usage.html
	     */
		static std::string backend;

	  public:
		//! Tuple object
		PyObject *pyEmptyTuple;

	  public:
		//! Annotate the point xy with text s
		PyObject *matplotlibFunction_annotate;
		//! Convenience method to get or set axis properties
		PyObject *matplotlibFunction_axis;
		//! Clear the current figure
		PyObject *matplotlibFunction_clf;
		//! Close the current figure
		PyObject *matplotlibFunction_close;
		//! Redraw the current figure
		PyObject *matplotlibFunction_draw;
		//! Plot y versus x as lines and/or markers with attached errorbars
		PyObject *matplotlibFunction_errorbar;
		//! Creates a new figure
		PyObject *matplotlibFunction_figure;
		//! Fill the area between two horizontal curves
		PyObject *matplotlibFunction_fill_between;
		//! Turn the axes grids on or off
		PyObject *matplotlibFunction_grid;
		//! Plot a histogram
		PyObject *matplotlibFunction_hist;
		//! Turn interactive mode on
		PyObject *matplotlibFunction_ion;
		//! Places a legend on the axes
		PyObject *matplotlibFunction_legend;
		//! Make a plot with log scaling on both the x and y axis
		PyObject *matplotlibFunction_loglog;
		//! Pause for interval seconds
		PyObject *matplotlibFunction_pause;
		//! Plot y versus x as lines and/or markers
		PyObject *matplotlibFunction_plot;
		//! Save the current figure
		PyObject *matplotlibFunction_savefig;
		//! Make a plot with log scaling on the x axis
		PyObject *matplotlibFunction_semilogx;
		//! Make a plot with log scaling on the y axis
		PyObject *matplotlibFunction_semilogy;
		//! Display the figure window
		PyObject *matplotlibFunction_show;
		//! Create a stem plot
		PyObject *matplotlibFunction_stem;
		//! Return a subplot axes at the given grid position
		PyObject *matplotlibFunction_subplot;
		//! Set a title of the current axes
		PyObject *matplotlibFunction_title;
		//! Automatically adjust subplot parameters to give specified padding
		PyObject *matplotlibFunction_tight_layout;
		//! Get or set the x limits of the current axes
		PyObject *matplotlibFunction_xlim;
		//! Set the x-axis label of the current axes
		PyObject *matplotlibFunction_xlabel;
		//! Turns on xkcd sketch-style drawing mode
		PyObject *matplotlibFunction_xkcd;
		//! Get or set the y limits of the current axes
		PyObject *matplotlibFunction_ylim;
		//! Set the y-axis label of the current axes.
		PyObject *matplotlibFunction_ylabel;
	};
};

/*!
 * To support all of use cases, matplotlib can target different outputs, and each of these capabilities is called a backend;
 * the “frontend” is the user facing code, i.e., the plotting code, whereas the “backend” does all the hard work
 * behind-the-scenes to make the figure.
 * There are two types of backends: user interface backends (for use in pygtk, wxpython, tkinter, qt4, or macosx; 
 * also referred to as “interactive backends”) and hardcopy backends to make image files (PNG, SVG, PDF, PS; also 
 * referred to as “non-interactive backends”).
 * 
 * Reference: 
 * https://matplotlib.org/tutorials/introductory/usage.html
 */
pyplot::matplotlib_interpreter::backend;

//! An instance of matplotlib_interpreter object
pyplot::matplotlib_interpreter pyplot::mpl;

/*!
 * \brief Annotate the point xy with text s
 * 
 * \tparam Data type
 * 
 * \param annotation  The text of the annotation
 * \param x           x point (x,y) to annotate
 * \param y           y point (x,y) to annotate
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::annotate(std::string const &annotation, T x, T y)
{
	return annotate<double>(annotation, static_cast<double>(x), static_cast<double>(y));
}

template <>
bool pyplot::annotate<double>(std::string const &annotation, double x, double y)
{
	PyObject *args = PyTuple_New(1);
	{
		PyObject *str = PyString_FromString(annotation.c_str());
		PyTuple_SetItem(args, 0, str);
	}

	//Create a new empty dictionary
	PyObject *kwargs = PyDict_New();
	{
		PyObject *xy = PyTuple_New(2);
		{
			PyTuple_SetItem(xy, 0, PyFloat_FromDouble(x));
			PyTuple_SetItem(xy, 1, PyFloat_FromDouble(y));
		}
		//Insert value into the dictionary kwargs using xy as a key
		PyDict_SetItemString(kwargs, "xy", xy);
	}

	PyObject *res = PyObject_Call(pyplot::mpl.matplotlibFunction_annotate, args, kwargs);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(kwargs);
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to annotate failed!" << std::endl;
	return false;
}

/*!
 * \brief Convenience method to get or set axis properties
 * 
 * \param axisArguments 
 */
bool pyplot::axis(std::string const &axisArguments)
{
	PyObject *args = PyTuple_New(1);
	{
		PyObject *str = PyString_FromString(axisArguments.c_str());
		PyTuple_SetItem(args, 0, str);
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_axis, args);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to axis failed!" << std::endl;
	return false;
}

/*!
 * \brief Clear the current figure
 * 
 */
inline bool pyplot::clf()
{
	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_clf, pyplot::mpl.pyEmptyTuple);

	if (res)
	{
		Py_DECREF(res);
		return true;
	}
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to clf failed!" << std::endl;
	return false;
}

/*!
 * \brief Close a figure window
 * 
 */
inline bool pyplot::close()
{
	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_close, pyplot::mpl.pyEmptyTuple);

	if (res)
	{
		Py_DECREF(res);
		return true;
	}
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to close failed!" << std::endl;
	return false;
}

/*!
 * \brief Redraw the current figure
 * This is used to update a figure that has been altered, 
 * but not automatically re-drawn. If interactive mode is 
 * on (ion()), this should be only rarely needed
 * 
 */
inline bool pyplot::draw()
{
	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_draw, pyplot::mpl.pyEmptyTuple);

	if (res)
	{
		Py_DECREF(res);
		return true;
	}
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to draw failed!" << std::endl;
	return false;
}

/*!
 * \brief Plot y versus x as lines and/or markers with attached errorbars
 * 
 * \tparam T
 * 
 * \param x       Scalar or array-like, data positions
 * \param y       Scalar or array-like, data positions
 * \param yerr    Errorbar
 * \param fmt     Plot format string
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::errorbar(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &yerr, std::string const &fmt = "")
{
	if (x.size() != y.size())
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Two vector should hav e the same size!" << std::endl;
		return false;
	}

	PyObject *args = PyTuple_New(4);
	{
		PyObject *xarray = PyArray<T>(x);
		PyObject *yarray = PyArray<T>(y);
		PyObject *yerrarray = PyArray<T>(yerr);
		PyObject *pystring = PyString_FromString(fmt.c_str());

		PyTuple_SetItem(args, 0, xarray);
		PyTuple_SetItem(args, 1, yarray);
		PyTuple_SetItem(args, 2, yerrarray);
		PyTuple_SetItem(args, 3, pystring);
	}

	PyObject *res = PyObject_Call(pyplot::mpl.matplotlibFunction_errorbar, args, pyplot::mpl.pyEmptyTuple);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}

	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to errorbar failed!" << std::endl;
	return false;
}

/*!
 * \brief Creates a new figure.
 * 
 */
inline bool pyplot::figure()
{
	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_figure, pyplot::mpl.pyEmptyTuple);

	if (res)
	{
		Py_DECREF(res);
		return true;
	}
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to figure failed!" << std::endl;
	return false;
}

/*!
 * \brief Creates a new figure
 * 
 * \param width   width in inches
 * \param height  height in inches
 * \param dpi     resolution of the figure (default is 100)
 * 
 * \return true 
 * \return false 
 */
bool pyplot::figure(std::size_t const width, std::size_t const height, std::size_t const dpi = 100)
{
	PyObject *kwargs = PyDict_New();
	{
		PyObject *size = PyTuple_New(2);
		{
			PyTuple_SetItem(size, 0, PyFloat_FromDouble(static_cast<double>(width) / static_cast<double>(dpi)));
			PyTuple_SetItem(size, 1, PyFloat_FromDouble(static_cast<double>(height) / static_cast<double>(dpi)));
		}
		PyDict_SetItemString(kwargs, "figsize", size);
		PyDict_SetItemString(kwargs, "dpi", PyLong_FromSize_t(dpi));
	}

	PyObject *res = PyObject_Call(pyplot::mpl.matplotlibFunction_figure, pyplot::mpl.pyEmptyTuple, kwargs);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		return true;
	}
	Py_DECREF(kwargs);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to figure failed!" << std::endl;
	return false;
}

/*!
 * \brief Fill the area between two horizontal curves.
 * The curves are defined by the points (x, y1) and (x, y2). 
 * This creates one or multiple polygons describing the filled area.
 * 
 * \tparam T 
 * 
 * \param x          The x coordinates of the nodes defining the curves.
 * \param y1         The y coordinates of the nodes defining the first curve.
 * \param y2         The y coordinates of the nodes defining the second curve.
 * \param keywords   All other keyword arguments are passed on to PolyCollection. They control the Polygon properties
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::fill_between(std::vector<T> const &x, std::vector<T> const &y1, std::vector<T> const &y2, std::map<std::string, std::string> const &keywords)
{
	if (x.size() != y1.size())
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Two vector should have the same size!" << std::endl;
		return false;
	}
	if (x.size() != y2.size())
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Two vector should have the same size!" << std::endl;
		return false;
	}

	// construct positional args
	PyObject *args = PyTuple_New(3);
	{
		// using numpy arrays
		PyObject *xarray = PyArray<T>(x);
		PyObject *y1array = PyArray<T>(y1);
		PyObject *y2array = PyArray<T>(y2);

		PyTuple_SetItem(args, 0, xarray);
		PyTuple_SetItem(args, 1, y1array);
		PyTuple_SetItem(args, 2, y2array);
	}

	// construct keyword args
	PyObject *kwargs = PyDict_New();
	for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
	{
		PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
	}

	PyObject *res = PyObject_Call(pyplot::mpl.matplotlibFunction_fill_between, args, kwargs);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		Py_DECREF(args);
		return true;
	}

	Py_DECREF(kwargs);
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to fill_between failed!" << std::endl;
	return false;
}

/*!
 * \brief Turn the axes grids on or off
 * 
 * \param flag 
 */
bool pyplot::grid(bool flag)
{
	PyObject *pyflag = flag ? Py_True : Py_False;
	Py_INCREF(pyflag);

	PyObject *args = PyTuple_New(1);
	PyTuple_SetItem(args, 0, pyflag);

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_grid, args);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}

	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to grid failed!" << std::endl;
	return false;
}

/*!
 * \brief Plot a histogram
 * Compute and draw the histogram of x. 
 * 
 * \tparam T 
 * 
 * \param x       Input values
 * \param bins    The bin specification (The default value is 10)
 * \param color   Color or array_like of colors or None, optional
 * \param alpha   The alpha blending value \f$ 0 <= scalar <= 1 \f$ or None, optional
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::hist(std::vector<T> const &x, long bins = 10, std::string const &color = "b", std::string const &label = "", double const alpha = 1.0)
{

	PyObject *args = PyTuple_New(1);
	{
		PyObject *xarray = PyArray<T>(x);
		PyTuple_SetItem(args, 0, xarray);
	}

	PyObject *kwargs = PyDict_New();
	{
		PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
		PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
		PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
		PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));
	}

	PyObject *res = PyObject_Call(pyplot::mpl.matplotlibFunction_hist, args, kwargs);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		Py_DECREF(args);
		return true;
	}

	Py_DECREF(kwargs);
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to hist failed!" << std::endl;
	return false;
}

/*!
 * \brief Turn interactive mode on
 * 
 */
inline bool pyplot::ion()
{
	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_ion, pyplot::mpl.pyEmptyTuple);

	if (res)
	{
		Py_DECREF(res);
		return true;
	}
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to ion failed!" << std::endl;
	return false;
}

/*!
 * \brief Places a legend on the axes
 * 
 */
inline bool pyplot::legend()
{
	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_legend, pyplot::mpl.pyEmptyTuple);

	if (res)
	{
		Py_DECREF(res);
		return true;
	}
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to legend failed!" << std::endl;
	return false;
}

/*!
 * \brief Make a plot with log scaling on both the x and y axis
 * This is just a thin wrapper around plot which additionally changes 
 * both the x-axis and the y-axis to log scaling. All of the concepts 
 * and parameters of plot can be used here as well.
 * 
 * \tparam T
 * 
 * \param x 
 * \param y 
 * \param fmt
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::loglog(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt = "", std::string const &label = "")
{
	if (x.size() != y1.size())
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Two vector should have the same size!" << std::endl;
		return false;
	}

	PyObject *args = PyTuple_New(3);
	{
		PyObject *xarray = PyArray<T>(x);
		PyObject *yarray = PyArray<T>(y);
		PyObject *pystring = PyString_FromString(fmt.c_str());

		PyTuple_SetItem(args, 0, xarray);
		PyTuple_SetItem(args, 1, yarray);
		PyTuple_SetItem(args, 2, pystring);
	}

	PyObject *kwargs = PyDict_New();
	{
		PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_loglog, args, kwargs);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(kwargs);
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to loglog failed!" << std::endl;
	return false;
}

/*!
 * \brief Pause for interval seconds.
 * 
 * \tparam T 
 * 
 * \param interval 
 */
bool pyplot::pause(double const interval)
{
	PyObject *args = PyTuple_New(1);
	PyTuple_SetItem(args, 0, PyFloat_FromDouble(interval));

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_pause, args);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}

	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to pause failed!" << std::endl;
	return false;
}

/*!
 * \brief Plot y versus x as lines and/or markers.
 * 
 * \tparam T 
 * 
 * \param x         array-like or scalar
 * \param y         array-like or scalar
 * \param keywords  are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::plot(std::vector<T> const &x, std::vector<T> const &y, std::map<std::string, std::string> const &keywords)
{
	if (x.size() != y.size())
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Two vector should hav e the same size!" << std::endl;
		throw std::runtime_error("Two vector of different sizes!");
	}

	//Construct positional args
	PyObject *args = PyTuple_New(2);
	{
		//Using numpy arrays
		PyObject *xarray = PyArray<T>(x);
		PyObject *yarray = PyArray<T>(y);

		PyTuple_SetItem(args, 0, xarray);
		PyTuple_SetItem(args, 1, yarray);
	}

	//Construct keyword args
	PyObject *kwargs = PyDict_New();
	for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
	{
		PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
	}

	PyObject *res = PyObject_Call(pyplot::mpl.matplotlibFunction_plot, args, kwargs);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		Py_DECREF(args);
		return true;
	}

	Py_DECREF(kwargs);
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to plot failed!" << std::endl;
	return false;
}

/*!
 * \brief Plot y versus x as lines and/or markers.
 * 
 * \tparam T 
 * 
 * \param x         array-like or scalar
 * \param y         array-like or scalar
 * \param fmt       A format string, e.g. ‘ro’ for red circles.
 *                  Format strings are just an abbreviation for quickly setting basic line properties. 
 * \param label     object. Set the label to s for auto legend
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::plot(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt = "", std::string const &label = "")
{
	if (x.size() != y.size())
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Two vector should hav e the same size!" << std::endl;
		throw std::runtime_error("Two vector of different sizes!");
	}

	//Construct positional args
	PyObject *args = PyTuple_New(3);
	{
		//Using numpy arrays
		PyObject *xarray = PyArray<T>(x);
		PyObject *yarray = PyArray<T>(y);
		PyObject *pystring = PyString_FromString(fmt.c_str());

		PyTuple_SetItem(args, 0, xarray);
		PyTuple_SetItem(args, 1, yarray);
		PyTuple_SetItem(args, 2, pystring);
	}

	PyObject *kwargs = PyDict_New();
	{
		PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_plot, args, kwargs);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		Py_DECREF(args);
		return true;
	}

	Py_DECREF(kwargs);
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to plot failed!" << std::endl;
	return false;
}

/*!
 * \brief Save the current figure
 * 
 * \param filename A string containing a path to a filename
 */
bool pyplot::savefig(std::string const &filename)
{
	PyObject *args = PyTuple_New(1);
	{
		PyObject *pyfilename = PyString_FromString(filename.c_str());
		PyTuple_SetItem(args, 0, pyfilename);
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_savefig, args);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}

	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to savefig failed!" << std::endl;
	return false;
}

/*!
 * \brief Make a plot with log scaling on the x axis
 * This is just a thin wrapper around plot which additionally changes the x-axis to log scaling. 
 * All of the concepts and parameters of plot can be used here as well.
 * 
 * \tparam T
 * 
 * \param x 
 * \param y 
 * \param fmt  
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::semilogx(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt = "", std::string const &label = "")
{
	if (x.size() != y.size())
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Two vector should hav e the same size!" << std::endl;
		throw std::runtime_error("Two vector of different sizes!");
	}

	PyObject *args = PyTuple_New(3);
	{
		PyObject *xarray = PyArray<T>(x);
		PyObject *yarray = PyArray<T>(y);
		PyObject *pystring = PyString_FromString(fmt.c_str());

		PyTuple_SetItem(plot_args, 0, xarray);
		PyTuple_SetItem(plot_args, 1, yarray);
		PyTuple_SetItem(plot_args, 2, pystring);
	}

	PyObject *kwargs = PyDict_New();
	{
		PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_semilogx, args, kwargs);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(kwargs);
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to semilogx failed!" << std::endl;
	return false;
}

/*!
 * \brief Make a plot with log scaling on the y axis.
 * 
 * \tparam T
 * 
 * \param x 
 * \param y 
 * \param fmt 
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::semilogy(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt = "", std::string const &label = "")
{
	if (x.size() != y.size())
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Two vector should hav e the same size!" << std::endl;
		throw std::runtime_error("Two vector of different sizes!");
	}

	PyObject *args = PyTuple_New(3);
	{
		PyObject *xarray = PyArray<T>(x);
		PyObject *yarray = PyArray<T>(y);
		PyObject *pystring = PyString_FromString(fmt.c_str());

		PyTuple_SetItem(plot_args, 0, xarray);
		PyTuple_SetItem(plot_args, 1, yarray);
		PyTuple_SetItem(plot_args, 2, pystring);
	}

	PyObject *kwargs = PyDict_New();
	{
		PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_semilogy, args, kwargs);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(kwargs);
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to semilogy failed!" << std::endl;
	return false;
}
/*!
 * \brief Display a figure. 
 * 
 * \param block 
 */
bool pyplot::show(bool const block = true)
{
	PyObject *res;
	if (block)
	{
		res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_show, pyplot::mpl.pyEmptyTuple);
	}
	else
	{
		PyObject *kwargs = PyDict_New();
		PyDict_SetItemString(kwargs, "block", Py_False);

		res = PyObject_Call(pyplot::mpl.matplotlibFunction_show, pyplot::mpl.pyEmptyTuple, kwargs);

		Py_DECREF(kwargs);
	}

	if (res)
	{
		Py_DECREF(res);
		return true;
	}

	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to show failed!" << std::endl;
	return false;
}

/*!
 * \brief Create a stem plot.
 * A stem plot plots vertical lines at each x location from the baseline to y, and places a marker there
 * 
 * \tparam T 
 * 
 * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1)
 * \param y         The y-values of the stem heads
 * \param keywords  
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::stem(std::vector<T> const &x, std::vector<T> const &y, std::map<std::string, std::string> const &keywords)
{
	if (x.size() != y.size())
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Two vector should hav e the same size!" << std::endl;
		throw std::runtime_error("Two vector of different sizes!");
	}

	//Construct positional args
	PyObject *args = PyTuple_New(2);
	{
		//Using numpy arrays
		PyObject *xarray = PyArray<T>(x);
		PyObject *yarray = PyArray<T>(y);

		PyTuple_SetItem(args, 0, xarray);
		PyTuple_SetItem(args, 1, yarray);
	}

	//Construct keyword args
	PyObject *kwargs = PyDict_New();
	for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
	{
		PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
	}

	PyObject *res = PyObject_Call(pyplot::mpl.matplotlibFunction_stem, args, kwargs);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		Py_DECREF(args);
		return true;
	}

	Py_DECREF(kwargs);
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to stem failed!" << std::endl;
	return false;
}

/*!
 * \brief Create a stem plot
 * A stem plot plots vertical lines at each x location from the baseline to y, and places a marker there.
 * 
 * \tparam T
 * 
 * \param x   The x-positions of the stems. Default: (0, 1, …, len(y) - 1).
 * \param y   The y-values of the stem heads.
 * \param fmt format
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::stem(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt = "")
{
	if (x.size() != y.size())
	{
		std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
		std::cerr << " Two vector should hav e the same size!" << std::endl;
		throw std::runtime_error("Two vector of different sizes!");
	}

	//Construct positional args
	PyObject *args = PyTuple_New(3);
	{
		//Using numpy arrays
		PyObject *xarray = PyArray<T>(x);
		PyObject *yarray = PyArray<T>(y);
		PyObject *pystring = PyString_FromString(fmt.c_str());

		PyTuple_SetItem(args, 0, xarray);
		PyTuple_SetItem(args, 1, yarray);
		PyTuple_SetItem(args, 2, pystring);
	}

	PyObject *res = PyObject_Call(pyplot::mpl.matplotlibFunction_stem, args);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to stem failed!" << std::endl;
	return false;
}

/*!
 * \brief Return a subplot axes at the given grid position
 * In the current figure, create and return an Axes, 
 * at position index of a (virtual) grid of nrows by ncols axes. 
 * Indexes go from 1 to nrows * ncols, incrementing in row-major order.
 * 
 * \param nrows 
 * \param ncols 
 * \param index 
 */
bool pyplot::subplot(long nrows, long ncols, long index)
{
	//Construct positional args
	PyObject *args = PyTuple_New(3);
	{
		PyTuple_SetItem(args, 0, PyFloat_FromDouble(nrows));
		PyTuple_SetItem(args, 1, PyFloat_FromDouble(ncols));
		PyTuple_SetItem(args, 2, PyFloat_FromDouble(index));
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_subplot, args);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to subplot failed!" << std::endl;
	return false;
}

/*!
 * \brief Set a title of the current axes
 * 
 * \param label Text to use for the title
 */
bool pyplot::title(std::string const &label)
{
	PyObject *args = PyTuple_New(1);
	{
		PyObject *pylabel = PyString_FromString(label.c_str());
		PyTuple_SetItem(args, 0, pylabel);
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_title, args);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to title failed!" << std::endl;
	return false;
}

// Actually, is there any reason not to call this automatically for every plot?
/*!
 * \brief Automatically adjust subplot parameters to give specified padding.
 * 
 */
inline bool pyplot::tight_layout()
{
	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_tight_layout, pyplot::mpl.pyEmptyTuple);

	if (res)
	{
		Py_DECREF(res);
		return true;
	}
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to tight_layout failed!" << std::endl;
	return false;
}

/*!
 * \brief Set the x limits of the current axes
 * 
 * \tparam T Data type
 * 
 * \param left  xmin
 * \param right xmax
 */
template <typename T>
bool pyplot::xlim(T left, T right)
{
	PyObject *args = PyTuple_New(1);
	{
		PyObject *list = PyList_New(2);
		PyList_SetItem(list, 0, PyFloat_FromDouble(left));
		PyList_SetItem(list, 1, PyFloat_FromDouble(right));

		PyTuple_SetItem(args, 0, list);
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_xlim, args);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to xlim failed!" << std::endl;
	return false;
}
/*!
 * \brief Get the x limits of the current axes
 * 
 * \tparam T Data type
 * 
 * \param left  xmin
 * \param right xmax
 */
template <typename T>
bool pyplot::xlim(T *left, T *right)
{
	PyObject *args = pyplot::mpl.pyEmptyTuple;
	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_xlim, args);
	PyObject *pleft = PyTuple_GetItem(res, 0);
	PyObject *pright = PyTuple_GetItem(res, 1);

	double arr[2];
	arr[0] = PyFloat_AsDouble(pleft);
	arr[1] = PyFloat_AsDouble(pright);

	*left = static_cast<T>(arr[0]);
	*right = static_cast<T>(arr[1]);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to xlim failed!" << std::endl;
	return false;
}

/*!
 * \brief Set the x-axis label of the current axes
 * 
 * \param label The label text
 */
bool pyplot::xlabel(std::string const &label)
{
	PyObject *args = PyTuple_New(1);
	{
		PyObject *pystr = PyString_FromString(label.c_str());
		PyTuple_SetItem(args, 0, pystr);
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_xlabel, args);
	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to xlabel failed!" << std::endl;
	return false;
}

/*!
 * \brief Turns on xkcd sketch-style drawing mode
 * This will only have effect on things drawn after this function is called
 * 
 */
inline bool pyplot::xkcd()
{
	PyObject *kwargs = PyDict_New();
	PyObject *res = PyObject_Call(pyplot::mpl.matplotlibFunction_xkcd, pyplot::mpl.pyEmptyTuple, kwargs);
	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(kwargs);
		return true;
	}
	Py_DECREF(kwargs);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to xkcd failed!" << std::endl;
	return false;
}

/*!
 * \brief Set the y limits of the current axes
 * 
 * \tparam T Data type
 * 
 * \param left  ymin
 * \param right ymax
 */
template <typename T>
bool pyplot::ylim(T left, T right)
{
	PyObject *args = PyTuple_New(1);
	{
		PyObject *list = PyList_New(2);
		PyList_SetItem(list, 0, PyFloat_FromDouble(left));
		PyList_SetItem(list, 1, PyFloat_FromDouble(right));

		PyTuple_SetItem(args, 0, list);
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_ylim, args);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to ylim failed!" << std::endl;
	return false;
}
/*!
 * \brief Get the y limits of the current axes
 * 
 * \tparam T Data type
 * 
 * \param left  ymin
 * \param right ymax
 */
template <typename T>
bool pyplot::ylim(T *left, T *right)
{
	PyObject *args = pyplot::mpl.pyEmptyTuple;
	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_ylim, args);
	PyObject *pleft = PyTuple_GetItem(res, 0);
	PyObject *pright = PyTuple_GetItem(res, 1);

	double arr[2];
	arr[0] = PyFloat_AsDouble(pleft);
	arr[1] = PyFloat_AsDouble(pright);

	*left = static_cast<T>(arr[0]);
	*right = static_cast<T>(arr[1]);

	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to ylim failed!" << std::endl;
	return false;
}

/*!
 * \brief Set the y-axis label of the current axes
 * 
 * \param label The label text
 */
bool pyplot::ylabel(std::string const &label)
{
	PyObject *args = PyTuple_New(1);
	{
		PyObject *pystr = PyString_FromString(label.c_str());
		PyTuple_SetItem(args, 0, pystr);
	}

	PyObject *res = PyObject_CallObject(pyplot::mpl.matplotlibFunction_ylabel, args);
	if (res)
	{
		Py_DECREF(res);
		Py_DECREF(args);
		return true;
	}
	Py_DECREF(args);
	std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
	std::cerr << " Call to ylabel failed!" << std::endl;
	return false;
}

#endif //HAVE_PYTHON
#endif //UMUQ_MATPLOTLIB_H
