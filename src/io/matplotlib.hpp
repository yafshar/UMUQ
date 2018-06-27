#ifndef UMUQ_MATPLOTLIB_H
#define UMUQ_MATPLOTLIB_H
#ifdef HAVE_PYTHON

/*!
 * \file io/matplotlib.hpp
 * \brief This module contains functions that allows to generate many kinds of plots
 *
 * The matplotlib Module contains addition, adaptation and modification to the 
 * original c++ interface to matplotlib source codes made available under the following LICENSE:
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
 * 
 * 
 * Matplotlib made available under the following LICENSE:
 * 
 * \verbatim
 * License agreement for matplotlib versions 1.3.0 and later
 * =========================================================
 * 
 * 1. This LICENSE AGREEMENT is between the Matplotlib Development Team
 * ("MDT"), and the Individual or Organization ("Licensee") accessing and
 * otherwise using matplotlib software in source or binary form and its
 * associated documentation.
 * 
 * 2. Subject to the terms and conditions of this License Agreement, MDT
 * hereby grants Licensee a nonexclusive, royalty-free, world-wide license
 * to reproduce, analyze, test, perform and/or display publicly, prepare
 * derivative works, distribute, and otherwise use matplotlib
 * alone or in any derivative version, provided, however, that MDT's
 * License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
 * 2012- Matplotlib Development Team; All Rights Reserved" are retained in
 * matplotlib  alone or in any derivative version prepared by
 * Licensee.
 * 
 * 3. In the event Licensee prepares a derivative work that is based on or
 * incorporates matplotlib or any part thereof, and wants to
 * make the derivative work available to others as provided herein, then
 * Licensee hereby agrees to include in any such work a brief summary of
 * the changes made to matplotlib .
 * 
 * 4. MDT is making matplotlib available to Licensee on an "AS
 * IS" basis.  MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
 * IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
 * DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
 * FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
 * WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.
 * 
 * 5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
 *  FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
 * LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
 * MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
 * THE POSSIBILITY THEREOF.
 * 
 * 6. This License Agreement will automatically terminate upon a material
 * breach of its terms and conditions.
 * 
 * 7. Nothing in this License Agreement shall be deemed to create any
 * relationship of agency, partnership, or joint venture between MDT and
 * Licensee.  This License Agreement does not grant permission to use MDT
 * trademarks or trade name in a trademark sense to endorse or promote
 * products or services of Licensee, or any third party.
 * 
 * 8. By copying, installing or otherwise using matplotlib ,
 * Licensee agrees to be bound by the terms and conditions of this License
 * Agreement.
 * \endverbatim
 */

#include "../misc/array.hpp"

/*!
 * \brief Type selector for numpy array conversion
 * 
 * \tparam T Data type
 */
template <typename T>
constexpr NPY_TYPES NPIDatatype = NPY_NOTYPE; // variable template
//bool
template <>
constexpr NPY_TYPES NPIDatatype<bool> = NPY_BOOL;
//int8
template <>
constexpr NPY_TYPES NPIDatatype<int8_t> = NPY_INT8;
template <>
constexpr NPY_TYPES NPIDatatype<uint8_t> = NPY_UINT8;
//short
template <>
constexpr NPY_TYPES NPIDatatype<int16_t> = NPY_SHORT;
template <>
constexpr NPY_TYPES NPIDatatype<uint16_t> = NPY_USHORT;
//int
template <>
constexpr NPY_TYPES NPIDatatype<int32_t> = NPY_INT;
template <>
constexpr NPY_TYPES NPIDatatype<uint32_t> = NPY_ULONG;
//int64
template <>
constexpr NPY_TYPES NPIDatatype<int64_t> = NPY_INT64;
template <>
constexpr NPY_TYPES NPIDatatype<uint64_t> = NPY_UINT64;
//float
template <>
constexpr NPY_TYPES NPIDatatype<float> = NPY_FLOAT;
template <>
constexpr NPY_TYPES NPIDatatype<double> = NPY_DOUBLE;
template <>
constexpr NPY_TYPES NPIDatatype<long double> = NPY_LONGDOUBLE;

/*!
 * \brief Converts a data array idata to Python array
 * 
 * \tparam T Data type
 * 
 * \param idata Input array of data
 * 
 * \return PyObject* Python array
 */
template <typename T>
PyObject *PyArray(std::vector<T> const &idata);

template <typename TIn, typename TOut>
PyObject *PyArray(std::vector<TIn> const &idata);

/*!
 * \brief Converts a data array idata to Python array
 * 
 * \tparam T Data type
 * 
 * \param idata array of data
 * \param nSize size of the array
 * \param Stride element stride (default is 1)
 * 
 * \return PyObject* Python array
 */
template <typename T>
PyObject *PyArray(T *idata, int const nSize, std::size_t const Stride = 1);

template <typename TIn, typename TOut>
PyObject *PyArray(TIn *idata, int const nSize, std::size_t const Stride = 1);

/*!
 * \verbatim
 * To support all of use cases, matplotlib can target different outputs, and each of these capabilities is called a backend;
 * the “frontend” is the user facing code, i.e., the plotting code, whereas the “backend” does all the hard work
 * behind-the-scenes to make the figure.
 * There are two types of backends: user interface backends (for use in pygtk, wxpython, tkinter, qt4, or macosx; 
 * also referred to as “interactive backends”) and hardcopy backends to make image files (PNG, SVG, PDF, PS; also 
 * referred to as “non-interactive backends”).
 * \endverbatim
 * 
 * Reference:
 * https://matplotlib.org/tutorials/introductory/usage.html
 */
static std::string backend;

/*!
 * \brief Set the “backend” to any of user interface backends
 * 
 * \param WXbackends user interface backends (for use in pygtk, wxpython, tkinter, qt4, or macosx; 
 *                   also referred to as “interactive backends”) or hardcopy backends to make image 
 *                   files (PNG, SVG, PDF, PS; also referred to as “non-interactive backends”)
 * 
 * NOTE: Must be called before the first regular call to matplotlib to have any effect
 * 
 * NOTE : Backend name specifications are not case-sensitive; e.g., ‘GTKAgg’ and ‘gtkagg’ are equivalent. 
 * 
 * Reference:
 * https://matplotlib.org/tutorials/introductory/usage.html
 * 
 * 
 * \verbatim
 * To make things a little more customizable for graphical user interfaces, matplotlib separates the concept 
 * of the renderer (the thing that actually does the drawing) from the canvas (the place where the drawing goes). 
 * The canonical renderer for user interfaces is Agg which uses the Anti-Grain Geometry C++ library to make a 
 * raster (pixel) image of the figure. All of the user interfaces except macosx can be used with agg rendering, 
 * e.g., WXAgg, GTKAgg, QT4Agg, QT5Agg, TkAgg. In addition, some of the user interfaces support other rendering engines. 
 * For example, with GTK, you can also select GDK rendering (backend GTK deprecated in 2.0) or Cairo rendering (backend GTKCairo).
 * 
 * 
 * Interactive backends, capable of displaying to the screen and of using appropriate 
 * renderers from the table above to write to a file include: 
 * ----------------------------------------------------------	
 * GTKAgg 	    Agg rendering to a GTK 2.x canvas (requires PyGTK and pycairo or cairocffi; Python2 only)
 * GTK3Agg 	    Agg rendering to a GTK 3.x canvas (requires PyGObject and pycairo or cairocffi)
 * GTK 	        GDK rendering to a GTK 2.x canvas (not recommended and d eprecated in 2.0) (requires PyGTK and pycairo or cairocffi; Python2 only)
 * GTKCairo 	Cairo rendering to a GTK 2.x canvas (requires PyGTK and pycairo or cairocffi; Python2 only)
 * GTK3Cairo 	Cairo rendering to a GTK 3.x canvas (requires PyGObject and pycairo or cairocffi)
 * WXAgg 	    Agg rendering to to a wxWidgets canvas (requires wxPython)
 * WX 	        Native wxWidgets drawing to a wxWidgets Canvas (not recommended and deprecated in 2.0) (requires wxPython)
 * TkAgg 	    Agg rendering to a Tk canvas (requires TkInter)
 * macosx 	    Cocoa rendering in OSX windows (presently lacks blocking show() behavior when matplotlib is in non-interactive mode)
 * ----------------------------------------------------------	
 * \endverbatim
 * 
 * Reference:
 * https://matplotlib.org/faq/usage_faq.html#wx-backends
 * 
 */
inline void setbackend(std::string const &WXbackends)
{
    backend = WXbackends;
}

/*! \class pyplot
 * \brief This module contains several common approaches to plotting with Matplotlib
 *
 * It contains below functions that allow you to generate many kinds of plots quickly:
 * 
 * \b annotate      Annotate the point xy with text s
 * \b axis          Convenience method to get or set axis properties
 * \b cla           Clear the current axis
 * \b clf           Clear the current figure
 * \b close         Close a figure window
 * \b draw          Redraw the current figure
 * \b errorbar      Plot y versus x as lines and/or markers with attached errorbars
 * \b figure        Creates a new figure
 * \b fill_between  Fill the area between two horizontal curves
 * \b grid          Turn the axes grids on or off
 * \b hist          Plot a histogram
 * \b ion           Turn interactive mode on
 * \b legend        Places a legend on the axes
 * \b loglog        Make a plot with log scaling on both the x and y axis
 * \b pause         Pause for interval seconds
 * \b plot          Plot y versus x as lines and/or markers
 * \b savefig       Save the current figure
 * \b scatter       A scatter plot of y vs x with varying marker size and/or color
 * \b semilogx      Make a plot with log scaling on the x axis
 * \b semilogy      Make a plot with log scaling on the y axis
 * \b show          Display a figure
 * \b stem          Create a stem plot
 * \b subplot       Return a subplot axes at the given grid position
 * \b title         Set a title of the current axes
 * \b tight_layout  Automatically adjust subplot parameters to give specified padding
 * \b xlim          Set/Get the x limits of the current axes
 * \b xlabel        Set the x-axis label of the current axes
 * \b xkcd          Turns on xkcd sketch-style drawing mode
 * \b ylim          Set/Get the y limits of the current axes
 * \b ylabel        Set the y-axis label of the current axes
 * 
 * Reference:
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
     * \brief Clear the current axes
     * 
     */
    inline bool cla();

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
    bool errorbar(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &yerr,
                  std::string const &fmt = "");

    /*!
     * \brief Plot y versus x as lines and/or markers with attached errorbars
     * 
     * \tparam T Data type
     * 
     * \param x         Scalar or array-like, data positions
     * \param nSizeX    Size of array x
     * \param StrideX   Stride element stride 
     * \param y         Scalar or array-like, data positions
     * \param nSizeY    Size of array y
     * \param StrideY   Stride element stride   
     * \param yerr      Scalar or array-like, data positions
     * \param nSizeE    Size of array
     * \param StrideE   Stride element stride 
     * \param fmt       Plot format string
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool errorbar(T const *x, int const nSizeX, std::size_t const StrideX,
                  T const *y, int const nSizeY, std::size_t const StrideY,
                  T const *yerr, int const nSizeE, std::size_t const StrideE,
                  std::string const &fmt = "");

    /*!
     * \brief Plot y versus x as lines and/or markers with attached errorbars
     * 
     * \tparam T Data type
     * 
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param yerr      Scalar or array-like, data positions
     * \param nSize     Size of array
     * \param fmt       Plot format string
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool errorbar(T const *x, T const *y, T const *yerr, int const nSize,
                  std::string const &fmt = "");

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
     * \brief Fill the area between two horizontal curves
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
    bool fill_between(std::vector<T> const &x, std::vector<T> const &y1, std::vector<T> const &y2,
                      std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Fill the area between two horizontal curves
     * The curves are defined by the points (x, y1) and (x, y2). 
     * This creates one or multiple polygons describing the filled area.
     * 
     * \tparam T         Data type
     * 
     * \param x          The x coordinates of the nodes defining the curves.
     * \param nSizeX     Size of array x
     * \param StrideX    Stride element stride 
     * \param y1         The y coordinates of the nodes defining the first curve.
     * \param nSizeY1    Size of array y1
     * \param StrideY1   Stride element stride  
     * \param y2         The y coordinates of the nodes defining the second curve.
     * \param nSizeY2    Size of array y2
     * \param StrideY2   Stride element stride   
     * \param keywords   All other keyword arguments are passed on to PolyCollection. They control the Polygon properties
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool fill_between(T const *x, int const nSizeX, std::size_t const StrideX,
                      T const *y1, int const nSizeY1, std::size_t const StrideY1,
                      T const *y2, int const nSizeY2, std::size_t const StrideY2,
                      std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Fill the area between two horizontal curves.
     * The curves are defined by the points (x, y1) and (x, y2). 
     * This creates one or multiple polygons describing the filled area.
     * 
     * \tparam T         Data type
     * 
     * \param x          The x coordinates of the nodes defining the curves.
     * \param y1         The y coordinates of the nodes defining the first curve.
     * \param y2         The y coordinates of the nodes defining the second curve.
     * \param nSize      Size of arrays  
     * \param keywords   All other keyword arguments are passed on to PolyCollection. They control the Polygon properties
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool fill_between(T const *x, T const *y1, T const *y2, int const nSize,
                      std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Turn the axes grids on or off
     * 
     * \param flag 
     */
    bool grid(bool flag);

    /*!
     * \brief Plot a histogram
     * Compute and draw the histogram of x
     * 
     * \tparam T 
     * 
     * \param x        Input values
     * \param bins     The bin specification (The default value is 50)
     * \param density  density (default false)
     *                 If True, the first element of the return tuple will be the counts 
     *                 normalized to form a probability density, i.e., the area (or integral) 
     *                 under the histogram will sum to 1. 
     *                 This is achieved by dividing the count by the number of observations 
     *                 times the bin width and not dividing by the total number of observations. 
     * \param color    Color or None, optional (The default value is "b")
     * \param label    default is None
     * \param alpha    The alpha blending value \f$ 0 <= scalar <= 1 \f$ or None, optional
     * \param Rcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
     * \param Gcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
     * \param Bcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool hist(std::vector<T> const &x, long const bins = 50, bool const density = false,
              std::string const &color = "b", std::string const &label = "", double const alpha = 1.0,
              int const Rcolor = 0, int const Gcolor = 0, int const Bcolor = 0);

    /*!
     * \brief Plot a histogram
     * Compute and draw the histogram of x
     * 
     * \tparam T       Data type
     * 
     * \param x        Input values
     * \param nSizeX   Size of array x
     * \param StrideX  Stride element stride 
     * \param bins     The bin specification (The default value is 50)
     * \param density  density (default false)
     *                 If True, the first element of the return tuple will be the counts 
     *                 normalized to form a probability density, i.e., the area (or integral) 
     *                 under the histogram will sum to 1. 
     *                 This is achieved by dividing the count by the number of observations 
     *                 times the bin width and not dividing by the total number of observations. 
     * \param color    Color or None, optional (The default value is "b")
     * \param label    default is None
     * \param alpha    The alpha blending value \f$ 0 <= scalar <= 1 \f$ or None, optional
     * \param Rcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
     * \param Gcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
     * \param Bcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool hist(T const *x, int const nSizeX, std::size_t const StrideX = 1,
              long const bins = 50, bool const density = false, std::string const &color = "b",
              std::string const &label = "", double const alpha = 1.0,
              int const Rcolor = 0, int const Gcolor = 0, int const Bcolor = 0);

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
     * \tparam T     Data type 
     * 
     * \param x      Scalar or array-like, data positions
     * \param y      Scalar or array-like, data positions
     * \param fmt    Plot format string
     * \param label  object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool loglog(std::vector<T> const &x, std::vector<T> const &y,
                std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Make a plot with log scaling on both the x and y axis
     * This is just a thin wrapper around plot which additionally changes 
     * both the x-axis and the y-axis to log scaling. All of the concepts 
     * and parameters of plot can be used here as well.
     * 
     * \tparam T        Data type 
     *  
     * \param x         Scalar or array-like, data positions
     * \param nSizeX    Size of array x
     * \param StrideX   Stride element stride 
     * \param y         Scalar or array-like, data positions
     * \param nSizeY    Size of array y
     * \param StrideY   Stride element stride 
     * \param fmt       Plot format string
     * \param label     object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool loglog(T const *x, int const nSizeX, std::size_t const StrideX,
                T const *y, int const nSizeY, std::size_t const StrideY,
                std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Make a plot with log scaling on both the x and y axis
     * This is just a thin wrapper around plot which additionally changes 
     * both the x-axis and the y-axis to log scaling. All of the concepts 
     * and parameters of plot can be used here as well.
     * 
     * \tparam T        Data type 
     *  
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param nSize     Size of array y
     * \param fmt       Plot format string
     * \param label     object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool loglog(T const *x, T const *y, int const nSize,
                std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Pause for interval seconds
     * 
     * \tparam T 
     * 
     * \param interval 
     */
    bool pause(double const interval);

    /*!
     * \brief Plot y versus x as lines and/or markers
     * 
     * \tparam T        Data type 
     * 
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool plot(std::vector<T> const &x, std::vector<T> const &y,
              std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Plot y versus x as lines and/or markers
     * 
     * \tparam T        Data type 
     *  
     * \param x         Scalar or array-like, data positions
     * \param nSizeX    Size of array x
     * \param StrideX   Stride element stride 
     * \param y         Scalar or array-like, data positions
     * \param nSizeY    Size of array y
     * \param StrideY   Stride element stride 
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool plot(T const *x, int const nSizeX, std::size_t const StrideX,
              T const *y, int const nSizeY, std::size_t const StrideY,
              std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Plot y versus x as lines and/or markers
     * 
     * \tparam T        Data type 
     *  
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param nSize     Size of arrays
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool plot(T const *x, T const *y, int const nSize,
              std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Plot y versus x as lines and/or markers
     * 
     * \tparam T        Data type 
     * 
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param fmt       A format string, e.g. ‘ro’ for red circles
     *                  Format strings are just an abbreviation for quickly setting basic line properties
     * \param label     object. Set the label to s for auto legend
     *  
     * \return true 
     * \return false 
     */
    template <typename T>
    bool plot(std::vector<T> const &x, std::vector<T> const &y,
              std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Plot y versus x as lines and/or markers
     * 
     * \tparam T        Data type 
     * 
     * \param x         Scalar or array-like, data positions
     * \param nSizeX    Size of array x
     * \param StrideX   Stride element stride 
     * \param y         Scalar or array-like, data positions
     * \param nSizeY    Size of array y
     * \param StrideY   Stride element stride 
     * \param fmt       A format string, e.g. ‘ro’ for red circles
     *                  Format strings are just an abbreviation for quickly setting basic line properties 
     * \param label     object. Set the label to s for auto legend
     *  
     * \return true 
     * \return false 
     */
    template <typename T>
    bool plot(T const *x, int const nSizeX, std::size_t const StrideX,
              T const *y, int const nSizeY, std::size_t const StrideY,
              std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Plot y versus x as lines and/or markers
     * 
     * \tparam T        Data type 
     * 
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param nSize     Size of arrays
     * \param fmt       A format string, e.g. ‘ro’ for red circles
     *                  Format strings are just an abbreviation for quickly setting basic line properties
     * \param label     object. Set the label to s for auto legend
     *  
     * \return true 
     * \return false 
     */
    template <typename T>
    bool plot(T const *x, T const *y, int const nSize,
              std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Save the current figure
     * 
     * \param filename A string containing a path to a filename
     */
    bool savefig(std::string const &filename);

    /*!
     * \brief A scatter plot of y vs x with varying marker size and/or color
     * 
     * \tparam T        Data type 
     * 
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param s         Scalar or array-like, marker size in points**2
     * \param c         Scalar or array-like, data color
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool scatter(std::vector<T> const &x, std::vector<T> const &y,
                 std::vector<T> const &s, std::vector<T> const &c,
                 std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief A scatter plot of y vs x with scaler marker size and color
     * 
     * \tparam T        Data type 
     * 
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param s         Scalar marker size in points**2
     * \param c         Scalar data color
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool scatter(std::vector<T> const &x, std::vector<T> const &y,
                 T const s, T const c,
                 std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief A scatter plot of y vs x with varying marker size and/or color
     * 
     * \tparam T        Data type 
     *  
     * \param x         Scalar or array-like, data positions
     * \param nSizeX    Size of array x
     * \param StrideX   Stride element stride 
     * \param y         Scalar or array-like, data positions
     * \param nSizeY    Size of array y
     * \param StrideY   Stride element stride 
     * \param s         Scalar or array-like, marker size in points**2
     * \param nSizeS    Size of array s
     * \param StrideS   Stride element stride 
     * \param c         Scalar or array-like, data colors
     * \param nSizeC    Size of array c
     * \param StrideC   Stride element stride 
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool scatter(T const *x, int const nSizeX, std::size_t const StrideX,
                 T const *y, int const nSizeY, std::size_t const StrideY,
                 T const *s, int const nSizeS, std::size_t const StrideS,
                 T const *c, int const nSizeC, std::size_t const StrideC,
                 std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief A scatter plot of y vs x with scaler marker size and color
     * 
     * \tparam T        Data type 
     *  
     * \param x         Scalar or array-like, data positions
     * \param nSizeX    Size of array x
     * \param StrideX   Stride element stride 
     * \param y         Scalar or array-like, data positions
     * \param nSizeY    Size of array y
     * \param StrideY   Stride element stride 
     * \param s         Scalar marker size in points**2
     * \param c         Scalar data colors
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool scatter(T const *x, int const nSizeX, std::size_t const StrideX,
                 T const *y, int const nSizeY, std::size_t const StrideY,
                 T const s, T const c,
                 std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief A scatter plot of y vs x with varying marker size and/or color
     * 
     * \tparam T        Data type 
     *  
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param s         Scalar or array-like, marker size in points**2
     * \param c         Scalar or array-like, data colors
     * \param nSize     Size of arrays
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool scatter(T const *x, T const *y, int const nSize,
                 T const *s, T const *c,
                 std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief A scatter plot of y vs x with scaler marker size and/or color
     * 
     * \tparam T        Data type 
     *  
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param nSize     Size of arrays
     * \param s         Scalar marker size in points**2
     * \param c         Scalar data color
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool scatter(T const *x, T const *y, int const nSize,
                 T const s, T const c,
                 std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Make a plot with log scaling on the x axis
     * This is just a thin wrapper around plot which additionally changes the x-axis to log scaling
     * All of the concepts and parameters of plot can be used here as well
     * 
     * \tparam T     Data type 
     * 
     * \param x      Scalar or array-like, data positions
     * \param y      Scalar or array-like, data positions
     * \param fmt    Plot format string
     * \param label  object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool semilogx(std::vector<T> const &x, std::vector<T> const &y,
                  std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Make a plot with log scaling on the x axis
     * This is just a thin wrapper around plot which additionally changes the x-axis to log scaling
     * All of the concepts and parameters of plot can be used here as well
     * 
     * \tparam T     Data type 
     * 
     * \param x         Scalar or array-like, data positions
     * \param nSizeX    Size of array x
     * \param StrideX   Stride element stride 
     * \param y         Scalar or array-like, data positions
     * \param nSizeY    Size of array y
     * \param StrideY   Stride element stride 
     * \param fmt    Plot format string
     * \param label  object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool semilogx(T const *x, int const nSizeX, std::size_t const StrideX,
                  T const *y, int const nSizeY, std::size_t const StrideY,
                  std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Make a plot with log scaling on the x axis
     * This is just a thin wrapper around plot which additionally changes the x-axis to log scaling.
     * All of the concepts and parameters of plot can be used here as well.
     * 
     * \tparam T     Data type 
     * 
     * \param x      Scalar or array-like, data positions
     * \param y      Scalar or array-like, data positions
     * \param nSize  Size of arrays
     * \param fmt    Plot format string
     * \param label  object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool semilogx(T const *x, T const *y, int const nSize,
                  std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Make a plot with log scaling on the y axis
     * This is just a thin wrapper around plot which additionally changes the y-axis to log scaling. 
     * All of the concepts and parameters of plot can be used here as well.
     * 
     * \tparam T     Data type 
     * 
     * \param x      Scalar or array-like, data positions
     * \param y      Scalar or array-like, data positions
     * \param fmt    Plot format string
     * \param label  object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool semilogy(std::vector<T> const &x, std::vector<T> const &y,
                  std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Make a plot with log scaling on the y axis
     * This is just a thin wrapper around plot which additionally changes the y-axis to log scaling. 
     * All of the concepts and parameters of plot can be used here as well.
     * 
     * \tparam T     Data type 
     * 
     * \param x         Scalar or array-like, data positions
     * \param nSizeX    Size of array x
     * \param StrideX   Stride element stride 
     * \param y         Scalar or array-like, data positions
     * \param nSizeY    Size of array y
     * \param StrideY   Stride element stride 
     * \param fmt    Plot format string
     * \param label  object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool semilogy(T const *x, int const nSizeX, std::size_t const StrideX,
                  T const *y, int const nSizeY, std::size_t const StrideY,
                  std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Make a plot with log scaling on the y axis
     * This is just a thin wrapper around plot which additionally changes the y-axis to log scaling. 
     * All of the concepts and parameters of plot can be used here as well.
     * 
     * \tparam T     Data type 
     * 
     * \param x      Scalar or array-like, data positions
     * \param y      Scalar or array-like, data positions
     * \param nSize  Size of arrays
     * \param fmt    Plot format string
     * \param label  object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool semilogy(T const *x, T const *y, int const nSize,
                  std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Display a figure
     * 
     * \param block 
     */
    bool show(bool const block = true);

    /*!
     * \brief Create a stem plot
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
    bool stem(std::vector<T> const &x, std::vector<T> const &y,
              std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Create a stem plot
     * A stem plot plots vertical lines at each x location from the baseline to y, and places a marker there
     * 
     * \tparam T        Data type 
     * 
     * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1)
     * \param nSizeX    Size of array x
     * \param StrideX   Stride element stride 
     * \param y         The y-values of the stem heads
     * \param nSizeY    Size of array y
     * \param StrideY   Stride element stride   
     * \param keywords  
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool stem(T const *x, int const nSizeX, std::size_t const StrideX,
              T const *y, int const nSizeY, std::size_t const StrideY,
              std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Create a stem plot
     * A stem plot plots vertical lines at each x location from the baseline to y, and places a marker there
     * 
     * \tparam T        Data type 
     * 
     * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1)
     * \param y         The y-values of the stem heads
     * \param nSize     Size of arrays   
     * \param keywords  
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool stem(T const *x, T const *y, int const nSize,
              std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Create a stem plot
     * A stem plot plots vertical lines at each x location from the baseline to y, and places a marker there.
     * 
     * \tparam T
     * 
     * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1).
     * \param y         The y-values of the stem heads.
     * \param fmt       A format string
     * \param label     object. Set the label to s for auto legend
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool stem(std::vector<T> const &x, std::vector<T> const &y,
              std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Create a stem plot
     * A stem plot plots vertical lines at each x location from the baseline to y, and places a marker there.
     * 
     * \tparam T        Data type 
     * 
     * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1)
     * \param nSizeX    Size of array x
     * \param StrideX   Stride element stride 
     * \param y         The y-values of the stem heads
     * \param nSizeY    Size of array y
     * \param StrideY   Stride element stride   
     * \param fmt       A format string
     * \param label     object. Set the label to s for auto legend
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool stem(T const *x, int const nSizeX, std::size_t const StrideX,
              T const *y, int const nSizeY, std::size_t const StrideY,
              std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Create a stem plot
     * A stem plot plots vertical lines at each x location from the baseline to y, and places a marker there.
     * 
     * \tparam T        Data type 
     * 
     * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1)
     * \param y         The y-values of the stem heads
     * \param nSize     Size of arrays  
     * \param fmt       A format string
     * \param label     object. Set the label to s for auto legend
     * 
     * \return true 
     * \return false 
     */
    template <typename T>
    bool stem(T const *x, T const *y, int const nSize,
              std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Return a subplot axes at the given grid position
     * In the current figure, create and return an Axes, at position index of a (virtual) grid of nrows by ncols axes. 
     * Indexes go from 1 to nrows * ncols, incrementing in row-major order.
     * 
     * \param nrows 
     * \param ncols 
     * \param index 
     */
    bool subplot(long const nrows, long const ncols, long const index);

    /*!
     * \brief Set a title of the current axes
     * 
     * \param label Text to use for the title
     */
    bool title(std::string const &label);

    /*!
     * \brief Automatically adjust subplot parameters to give specified padding
     * 
     * TOCHECK:
     * NOTE: We should call this automatically for every plot!
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

  private:
    // Make it noncopyable
    pyplot(pyplot const &) = delete;

    // Make it not assignable
    pyplot &operator=(pyplot const &) = delete;

  private:
    /*! \class matplotlib
     * \brief This class sets and initializes python matplotlib for different use cases
     * 
     * \verbatim
     * To support all of use cases, matplotlib can target different outputs, and each of these capabilities is called a backend
     * the “frontend” is the user facing code, i.e., the plotting code, whereas the “backend” does all the hard work
     * behind-the-scenes to make the figure.
     * \endverbatim
     * 
     * Reference:
     * https://matplotlib.org
     * 
     */
    class matplotlib
    {
      public:
        /*!
         * \brief Construct a new matplotlib interpreter object
         * 
         */
        matplotlib();

        /*!
         * \brief Destroy the matplotlib interpreter object
         * 
         */
        ~matplotlib()
        {
            // Undo all initializations made by Py_Initialize() and subsequent use
            // of Python/C API functions, and destroy all sub-interpreters
            Py_Finalize();
        }

      public:
        // Make it noncopyable
        matplotlib(matplotlib const &) = delete;

        // Make it not assignable
        matplotlib &operator=(matplotlib const &) = delete;

      public:
        //! Tuple object
        PyObject *pyEmpty;

      public:
        //! Annotate the point xy with text s
        PyObject *pyannotate;
        //! Convenience method to get or set axis properties
        PyObject *pyaxis;
        //! Clear the current axes
        PyObject *pycla;
        //! Clear the current figure
        PyObject *pyclf;
        //! Close the current figure
        PyObject *pyclose;
        //! Redraw the current figure
        PyObject *pydraw;
        //! Plot y versus x as lines and/or markers with attached errorbars
        PyObject *pyerrorbar;
        //! Creates a new figure
        PyObject *pyfigure;
        //! Fill the area between two horizontal curves
        PyObject *pyfill_between;
        //! Turn the axes grids on or off
        PyObject *pygrid;
        //! Plot a histogram
        PyObject *pyhist;
        //! Turn interactive mode on
        PyObject *pyion;
        //! Places a legend on the axes
        PyObject *pylegend;
        //! Make a plot with log scaling on both the x and y axis
        PyObject *pyloglog;
        //! Pause for interval seconds
        PyObject *pypause;
        //! Plot y versus x as lines and/or markers
        PyObject *pyplot;
        //! Save the current figure
        PyObject *pysavefig;
        //! A scatter plot of y vs x with varying marker size and/or color
        PyObject *pyscatter;
        //! Make a plot with log scaling on the x axis
        PyObject *pysemilogx;
        //! Make a plot with log scaling on the y axis
        PyObject *pysemilogy;
        //! Display the figure window
        PyObject *pyshow;
        //! Create a stem plot
        PyObject *pystem;
        //! Return a subplot axes at the given grid position
        PyObject *pysubplot;
        //! Set a title of the current axes
        PyObject *pytitle;
        //! Automatically adjust subplot parameters to give specified padding
        PyObject *pytight_layout;
        //! Get or set the x limits of the current axes
        PyObject *pyxlim;
        //! Set the x-axis label of the current axes
        PyObject *pyxlabel;
        //! Turns on xkcd sketch-style drawing mode
        PyObject *pyxkcd;
        //! Get or set the y limits of the current axes
        PyObject *pyylim;
        //! Set the y-axis label of the current axes.
        PyObject *pyylabel;
    };

  public:
    //! An instance of matplotlib object
    static matplotlib mpl;
};

//! An instance of matplotlib object
pyplot::matplotlib pyplot::mpl;

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

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyObject *xy = PyTuple_New(2);
        {
            PyTuple_SetItem(xy, 0, PyFloat_FromDouble(x));
            PyTuple_SetItem(xy, 1, PyFloat_FromDouble(y));
        }
        //Insert value into the dictionary kwargs using xy as a key
        PyDict_SetItemString(kwargs, "xy", xy);
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyannotate, args, kwargs);

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

    PyObject *res = PyObject_CallObject(pyplot::mpl.pyaxis, args);

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
 * \brief Clear the current axis
 * 
 */
inline bool pyplot::cla()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pycla, pyplot::mpl.pyEmpty);

    if (res)
    {
        Py_DECREF(res);
        return true;
    }
    std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
    std::cerr << " Call to cla failed!" << std::endl;
    return false;
}

/*!
 * \brief Clear the current figure
 * 
 */
inline bool pyplot::clf()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyclf, pyplot::mpl.pyEmpty);

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
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyclose, pyplot::mpl.pyEmpty);

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
    PyObject *res = PyObject_CallObject(pyplot::mpl.pydraw, pyplot::mpl.pyEmpty);

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
bool pyplot::errorbar(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &yerr, std::string const &fmt)
{
    if (x.size() != y.size() || x.size() != yerr.size())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
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

    PyObject *res = PyObject_Call(pyplot::mpl.pyerrorbar, args, pyplot::mpl.pyEmpty);

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
 * \brief Plot y versus x as lines and/or markers with attached errorbars
 * 
 * \tparam T Data type
 * 
 * \param x         Scalar or array-like, data positions
 * \param nSizeX    Size of array x
 * \param StrideX   Stride element stride 
 * \param y         Scalar or array-like, data positions
 * \param nSizeY    Size of array y
 * \param StrideY   Stride element stride   
 * \param yerr      Scalar or array-like, data positions
 * \param nSizeE    Size of array
 * \param StrideE   Stride element stride 
 * \param fmt       Plot format string
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::errorbar(T const *x, int const nSizeX, std::size_t const StrideX,
                      T const *y, int const nSizeY, std::size_t const StrideY,
                      T const *yerr, int const nSizeE, std::size_t const StrideE,
                      std::string const &fmt)
{
    {
        std::size_t nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
        std::size_t nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;
        std::size_t nsizeE = StrideE == 1 ? nSizeE : nSizeE / StrideE;

        if (nsizeX != nsizeY || nsizeX != nsizeE)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    PyObject *args = PyTuple_New(4);
    {
        PyObject *xarray = PyArray<T>(x, nSizeX, StrideX);
        PyObject *yarray = PyArray<T>(y, nSizeY, StrideY);
        PyObject *yerrarray = PyArray<T>(yerr, nSizeE, StrideE);

        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, yerrarray);
        PyTuple_SetItem(args, 3, pystring);
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyerrorbar, args, pyplot::mpl.pyEmpty);

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
 * \brief Plot y versus x as lines and/or markers with attached errorbars
 * 
 * \tparam T Data type
 * 
 * \param x         Scalar or array-like, data positions
 * \param y         Scalar or array-like, data positions
 * \param yerr      Scalar or array-like, data positions
 * \param nSize     Size of array
 * \param fmt       Plot format string
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::errorbar(T const *x, T const *y, T const *yerr, int const nSize, std::string const &fmt)
{
    PyObject *args = PyTuple_New(4);
    {
        PyObject *xarray = PyArray<T>(x, nSize);
        PyObject *yarray = PyArray<T>(y, nSize);
        PyObject *yerrarray = PyArray<T>(yerr, nSize);

        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, yerrarray);
        PyTuple_SetItem(args, 3, pystring);
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyerrorbar, args, pyplot::mpl.pyEmpty);

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
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyfigure, pyplot::mpl.pyEmpty);

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
bool pyplot::figure(std::size_t const width, std::size_t const height, std::size_t const dpi)
{
    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyObject *size = PyTuple_New(2);
        {
            PyTuple_SetItem(size, 0, PyFloat_FromDouble(static_cast<double>(width) / static_cast<double>(dpi)));
            PyTuple_SetItem(size, 1, PyFloat_FromDouble(static_cast<double>(height) / static_cast<double>(dpi)));
        }
        PyDict_SetItemString(kwargs, "figsize", size);
        PyDict_SetItemString(kwargs, "dpi", PyLong_FromSize_t(dpi));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyfigure, pyplot::mpl.pyEmpty, kwargs);

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
 * \tparam T         Data type
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
    if (x.size() != y1.size() || x.size() != y2.size())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
        return false;
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *y1array = PyArray<T>(y1);
        PyObject *y2array = PyArray<T>(y2);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, y1array);
        PyTuple_SetItem(args, 2, y2array);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyfill_between, args, kwargs);

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
 * \brief Fill the area between two horizontal curves.
 * The curves are defined by the points (x, y1) and (x, y2). 
 * This creates one or multiple polygons describing the filled area.
 * 
 * \tparam T         Data type
 * 
 * \param x          The x coordinates of the nodes defining the curves.
 * \param nSizeX     Size of array x
 * \param StrideX    Stride element stride 
 * \param y1         The y coordinates of the nodes defining the first curve.
 * \param nSizeY1    Size of array y1
 * \param StrideY1   Stride element stride  
 * \param y2         The y coordinates of the nodes defining the second curve.
 * \param nSizeY2    Size of array y2
 * \param StrideY2   Stride element stride   
 * \param keywords   All other keyword arguments are passed on to PolyCollection. They control the Polygon properties
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::fill_between(T const *x, int const nSizeX, std::size_t const StrideX,
                          T const *y1, int const nSizeY1, std::size_t const StrideY1,
                          T const *y2, int const nSizeY2, std::size_t const StrideY2,
                          std::map<std::string, std::string> const &keywords)
{
    {
        std::size_t nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
        std::size_t nsizeY1 = StrideY1 == 1 ? nSizeY1 : nSizeY1 / StrideY1;
        std::size_t nsizeY2 = StrideY2 == 1 ? nSizeY2 : nSizeY2 / StrideY2;

        if (nsizeX != nsizeY1 || nsizeX != nsizeY2)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSizeX, StrideX);
        PyObject *y1array = PyArray<T>(y1, nSizeY1, StrideY1);
        PyObject *y2array = PyArray<T>(y2, nSizeY2, StrideY2);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, y1array);
        PyTuple_SetItem(args, 2, y2array);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyfill_between, args, kwargs);

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
 * \brief Fill the area between two horizontal curves.
 * The curves are defined by the points (x, y1) and (x, y2). 
 * This creates one or multiple polygons describing the filled area.
 * 
 * \tparam T         Data type
 * 
 * \param x          The x coordinates of the nodes defining the curves.
 * \param y1         The y coordinates of the nodes defining the first curve.
 * \param y2         The y coordinates of the nodes defining the second curve.
 * \param nSize      Size of arrays  
 * \param keywords   All other keyword arguments are passed on to PolyCollection. They control the Polygon properties
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::fill_between(T const *x, T const *y1, T const *y2, int const nSize, std::map<std::string, std::string> const &keywords)
{
    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSize);
        PyObject *y1array = PyArray<T>(y1, nSize);
        PyObject *y2array = PyArray<T>(y2, nSize);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, y1array);
        PyTuple_SetItem(args, 2, y2array);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyfill_between, args, kwargs);

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
    PyObject *pyflag;
    PyObject *args = PyTuple_New(1);
    {
        pyflag = flag ? Py_True : Py_False;
        Py_INCREF(pyflag);
        PyTuple_SetItem(args, 0, pyflag);
    }

    PyObject *res = PyObject_CallObject(pyplot::mpl.pygrid, args);

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
 * \tparam T       Data type
 * 
 * \param x        Input values
 * \param bins     The bin specification (The default value is 10)
 * \param density  density (Default is false)
 *                 If True, the first element of the return tuple will be the counts 
 *                 normalized to form a probability density, i.e., the area (or integral) 
 *                 under the histogram will sum to 1. 
 *                 This is achieved by dividing the count by the number of observations 
 *                 times the bin width and not dividing by the total number of observations. 
 * \param color    Color or None, optional (The default value is "b")
 * \param label    default is None
 * \param alpha    The alpha blending value \f$ 0 <= scalar <= 1 \f$ or None, optional
 * \param Rcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
 * \param Gcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
 * \param Bcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::hist(std::vector<T> const &x, long const bins, bool const density,
                  std::string const &color, std::string const &label, double const alpha,
                  int const Rcolor, int const Gcolor, int const Bcolor)
{
    PyObject *args = PyTuple_New(2);
    {
        PyObject *xarray = PyArray<T>(x);
        PyObject *pybins = PyInt_FromLong(bins);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, pybins);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    PyObject *pyColor = NULL;
    {
        // Construct keyword args
        if (density)
        {
            PyDict_SetItemString(kwargs, "normed", Py_True);
        }
        if (Rcolor != 0 || Gcolor != 0 || Bcolor != 0)
        {
            pyColor = PyTuple_New(3);
            {
                PyObject *pyRcolor = PyFloat_FromDouble(static_cast<double>(Rcolor) / 255.);
                PyObject *pyGcolor = PyFloat_FromDouble(static_cast<double>(Gcolor) / 255.);
                PyObject *pyBcolor = PyFloat_FromDouble(static_cast<double>(Bcolor) / 255.);

                PyTuple_SetItem(pyColor, 0, pyRcolor);
                PyTuple_SetItem(pyColor, 1, pyGcolor);
                PyTuple_SetItem(pyColor, 2, pyBcolor);
            }

            PyDict_SetItemString(kwargs, "color", pyColor);
        }
        else
        {
            PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
        }
        PyDict_SetItemString(kwargs, "ec", PyString_FromString("black"));
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
        PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyhist, args, kwargs);

    if (pyColor)
    {
        Py_DECREF(pyColor);
    }
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
 * \brief Plot a histogram
 * Compute and draw the histogram of x. 
 * 
 * \tparam T       Data type
 * 
 * \param x        Input values
 * \param nSizeX   Size of array x
 * \param StrideX  Stride element stride 
 * \param bins     The bin specification (The default value is 10)
 * \param density  density (default false)
 *                 If True, the first element of the return tuple will be the counts 
 *                 normalized to form a probability density, i.e., the area (or integral) 
 *                 under the histogram will sum to 1. 
 *                 This is achieved by dividing the count by the number of observations 
 *                 times the bin width and not dividing by the total number of observations. 
 * \param color    Color or None, optional (The default value is "b")
 * \param label    default is None
 * \param alpha    The alpha blending value \f$ 0 <= scalar <= 1 \f$ or None, optional
 * \param Rcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
 * \param Gcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
 * \param Bcolor   Color of array_like colors (Rcolor/255, Gcolor/255, Bcolor/255), optional (The default value is 0)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::hist(T const *x, int const nSizeX, std::size_t const StrideX,
                  long const bins, bool const density, std::string const &color,
                  std::string const &label, double const alpha,
                  int const Rcolor, int const Gcolor, int const Bcolor)
{
    PyObject *args = PyTuple_New(2);
    {
        PyObject *xarray = PyArray<T>(x, nSizeX, StrideX);
        PyObject *pybins = PyInt_FromLong(bins);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, pybins);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    PyObject *pyColor = NULL;
    {
        // Construct keyword args
        if (density)
        {
            PyDict_SetItemString(kwargs, "normed", Py_True);
        }
        if (Rcolor != 0 || Gcolor != 0 || Bcolor != 0)
        {
            pyColor = PyTuple_New(3);
            {
                PyObject *pyRcolor = PyFloat_FromDouble(static_cast<double>(Rcolor) / 255.);
                PyObject *pyGcolor = PyFloat_FromDouble(static_cast<double>(Gcolor) / 255.);
                PyObject *pyBcolor = PyFloat_FromDouble(static_cast<double>(Bcolor) / 255.);

                PyTuple_SetItem(pyColor, 0, pyRcolor);
                PyTuple_SetItem(pyColor, 1, pyGcolor);
                PyTuple_SetItem(pyColor, 2, pyBcolor);
            }

            PyDict_SetItemString(kwargs, "color", pyColor);
        }
        else
        {
            PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
        }
        PyDict_SetItemString(kwargs, "ec", PyString_FromString("black"));
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
        PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyhist, args, kwargs);

    if (pyColor)
    {
        Py_DECREF(pyColor);
    }
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
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyion, pyplot::mpl.pyEmpty);

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
    PyObject *res = PyObject_CallObject(pyplot::mpl.pylegend, pyplot::mpl.pyEmpty);

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
 * \tparam T     Data type
 * 
 * \param x      Input values
 * \param y      Input values
 * \param fmt    Plot format string
 * \param label  object (Set the label to s for auto legend)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::loglog(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt, std::string const &label)
{
    if (x.size() != y.size())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
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

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyloglog, args, kwargs);

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
 * \brief Make a plot with log scaling on both the x and y axis
 * This is just a thin wrapper around plot which additionally changes 
 * both the x-axis and the y-axis to log scaling. All of the concepts 
 * and parameters of plot can be used here as well.
 * 
 * \tparam T        Data type 
 *  
 * \param x         Scalar or array-like, data positions
 * \param nSizeX    Size of array x
 * \param StrideX   Stride element stride 
 * \param y         Scalar or array-like, data positions
 * \param nSizeY    Size of array y
 * \param StrideY   Stride element stride 
 * \param fmt       Plot format string
 * \param label     object (Set the label to s for auto legend)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::loglog(T const *x, int const nSizeX, std::size_t const StrideX,
                    T const *y, int const nSizeY, std::size_t const StrideY,
                    std::string const &fmt, std::string const &label)
{
    {
        std::size_t nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
        std::size_t nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;

        if (nsizeX != nsizeY)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    PyObject *args = PyTuple_New(3);
    {
        PyObject *xarray = PyArray<T>(x, nSizeX, StrideX);
        PyObject *yarray = PyArray<T>(y, nSizeY, StrideY);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyloglog, args, kwargs);

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
 * \brief Make a plot with log scaling on both the x and y axis
 * This is just a thin wrapper around plot which additionally changes 
 * both the x-axis and the y-axis to log scaling. All of the concepts 
 * and parameters of plot can be used here as well.
 * 
 * \tparam T        Data type 
 *  
 * \param x         Scalar or array-like, data positions
 * \param y         Scalar or array-like, data positions
 * \param nSize     Size of array y
 * \param fmt       Plot format string
 * \param label     object (Set the label to s for auto legend)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::loglog(T const *x, T const *y, int const nSize, std::string const &fmt, std::string const &label)
{
    PyObject *args = PyTuple_New(3);
    {
        PyObject *xarray = PyArray<T>(x, nSize);
        PyObject *yarray = PyArray<T>(y, nSize);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyloglog, args, kwargs);

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
    {
        PyTuple_SetItem(args, 0, PyFloat_FromDouble(interval));
    }

    PyObject *res = PyObject_CallObject(pyplot::mpl.pypause, args);

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
 * \tparam T        Data type 
 * 
 * \param x         Scalar or array-like, data positions
 * \param y         Scalar or array-like, data positions
 * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
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
        std::cerr << "Two vectors should have the same size!" << std::endl;
        return false;
    }

    // Construct positional args
    PyObject *args = PyTuple_New(2);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *yarray = PyArray<T>(y);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyplot, args, kwargs);

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
 * \tparam T        Data type 
 *  
 * \param x         Scalar or array-like, data positions
 * \param nSizeX    Size of array x
 * \param StrideX   Stride element stride 
 * \param y         Scalar or array-like, data positions
 * \param nSizeY    Size of array y
 * \param StrideY   Stride element stride 
 * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::plot(T const *x, int const nSizeX, std::size_t const StrideX,
                  T const *y, int const nSizeY, std::size_t const StrideY,
                  std::map<std::string, std::string> const &keywords)
{
    {
        std::size_t nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
        std::size_t nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;

        if (nsizeX != nsizeY)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    // Construct positional args
    PyObject *args = PyTuple_New(2);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSizeX, StrideX);
        PyObject *yarray = PyArray<T>(y, nSizeY, StrideY);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyplot, args, kwargs);

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
 * \tparam T        Data type 
 *  
 * \param x         Scalar or array-like, data positions
 * \param y         Scalar or array-like, data positions
 * \param nSize     Size of arrays
 * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::plot(T const *x, T const *y, int const nSize, std::map<std::string, std::string> const &keywords)
{
    // Construct positional args
    PyObject *args = PyTuple_New(2);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSize);
        PyObject *yarray = PyArray<T>(y, nSize);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyplot, args, kwargs);

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
 * \tparam T        Data type 
 * 
 * \param x         Scalar or array-like, data positions
 * \param y         Scalar or array-like, data positions
 * \param fmt       A format string, e.g. ‘ro’ for red circles.
 *                  Format strings are just an abbreviation for quickly setting basic line properties. 
 * \param label     object. Set the label to s for auto legend
 *  
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::plot(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt, std::string const &label)
{
    if (x.size() != y.size())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
        return false;
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *yarray = PyArray<T>(y);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyplot, args, kwargs);

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
 * \tparam T        Data type 
 * 
 * \param x         Scalar or array-like, data positions
 * \param nSizeX    Size of array x
 * \param StrideX   Stride element stride 
 * \param y         Scalar or array-like, data positions
 * \param nSizeY    Size of array y
 * \param StrideY   Stride element stride 
 * \param fmt       A format string, e.g. ‘ro’ for red circles.
 *                  Format strings are just an abbreviation for quickly setting basic line properties. 
 * \param label     object. Set the label to s for auto legend
 *  
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::plot(T const *x, int const nSizeX, std::size_t const StrideX,
                  T const *y, int const nSizeY, std::size_t const StrideY,
                  std::string const &fmt, std::string const &label)
{
    {
        std::size_t nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
        std::size_t nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;

        if (nsizeX != nsizeY)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSizeX, StrideX);
        PyObject *yarray = PyArray<T>(y, nSizeY, StrideY);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyplot, args, kwargs);

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
 * \tparam T        Data type 
 * 
 * \param x         Scalar or array-like, data positions
 * \param y         Scalar or array-like, data positions
 * \param nSize     Size of arrays
 * \param fmt       A format string, e.g. ‘ro’ for red circles.
 *                  Format strings are just an abbreviation for quickly setting basic line properties. 
 * \param label     object. Set the label to s for auto legend
 *  
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::plot(T const *x, T const *y, int const nSize, std::string const &fmt, std::string const &label)
{
    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSize);
        PyObject *yarray = PyArray<T>(y, nSize);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyplot, args, kwargs);

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

    PyObject *res = PyObject_CallObject(pyplot::mpl.pysavefig, args);

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
 * \brief A scatter plot of y vs x with varying marker size and/or color
 * 
 * \tparam T        Data type 
 * 
 * \param x         Scalar or array-like, data positions
 * \param y         Scalar or array-like, data positions
 * \param s         Scalar or array-like, marker size in points**2
 * \param c         Scalar or array-like, data color
 * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::scatter(std::vector<T> const &x, std::vector<T> const &y,
                     std::vector<T> const &s, std::vector<T> const &c,
                     std::map<std::string, std::string> const &keywords)
{
    if (x.size() != y.size())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
        return false;
    }
    if (s.size() > 1)
    {
        if (x.size() != s.size())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }
    if (c.size() > 1)
    {
        if (x.size() != c.size())
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x);
        PyObject *PyArrayY = PyArray<T>(y);
        PyObject *PyArrayS = s.size() > 1 ? PyArray<T>(s) : PyFloat_FromDouble(static_cast<double>(s[0]));
        PyObject *PyArrayC = c.size() > 1 ? PyArray<T>(c) : PyFloat_FromDouble(static_cast<double>(c[0]));

        PyTuple_SetItem(args, 0, PyArrayX);
        PyTuple_SetItem(args, 1, PyArrayY);
        PyTuple_SetItem(args, 2, PyArrayS);
        PyTuple_SetItem(args, 3, PyArrayC);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // No patch boundary will be drawn
        // For non-filled markers, the edgecolors kwarg is ignored and forced to ‘face’ internally.
        PyDict_SetItemString(kwargs, "edgecolor", PyString_FromString("None"));
    }
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyscatter, args, kwargs);

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
    std::cerr << " Call to scatter failed!" << std::endl;
    return false;
}

/*!
 * \brief A scatter plot of y vs x with scaler marker size and color
 * 
 * \tparam T        Data type 
 * 
 * \param x         Scalar or array-like, data positions
 * \param y         Scalar or array-like, data positions
 * \param s         Scalar marker size in points**2
 * \param c         Scalar data color
 * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::scatter(std::vector<T> const &x, std::vector<T> const &y,
                     T const s, T const c,
                     std::map<std::string, std::string> const &keywords)
{
    if (x.size() != y.size())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
        return false;
    }

    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x);
        PyObject *PyArrayY = PyArray<T>(y);
        PyObject *PyArrayS = PyFloat_FromDouble(static_cast<double>(s));
        PyObject *PyArrayC = PyFloat_FromDouble(static_cast<double>(c));

        PyTuple_SetItem(args, 0, PyArrayX);
        PyTuple_SetItem(args, 1, PyArrayY);
        PyTuple_SetItem(args, 2, PyArrayS);
        PyTuple_SetItem(args, 3, PyArrayC);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // No patch boundary will be drawn
        // For non-filled markers, the edgecolors kwarg is ignored and forced to ‘face’ internally.
        PyDict_SetItemString(kwargs, "edgecolor", PyString_FromString("None"));
    }
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyscatter, args, kwargs);

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
    std::cerr << " Call to scatter failed!" << std::endl;
    return false;
}

/*!
 * \brief A scatter plot of y vs x with varying marker size and/or color
 * 
 * \tparam T        Data type 
 *  
 * \param x         Scalar or array-like, data positions
 * \param nSizeX    Size of array x
 * \param StrideX   Stride element stride 
 * \param y         Scalar or array-like, data positions
 * \param nSizeY    Size of array y
 * \param StrideY   Stride element stride 
 * \param s         Scalar or array-like, marker size in points**2
 * \param nSizeS    Size of array s
 * \param StrideS   Stride element stride 
 * \param c         Scalar or array-like, data colors
 * \param nSizeC    Size of array c
 * \param StrideC   Stride element stride 
 * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::scatter(T const *x, int const nSizeX, std::size_t const StrideX,
                     T const *y, int const nSizeY, std::size_t const StrideY,
                     T const *s, int const nSizeS, std::size_t const StrideS,
                     T const *c, int const nSizeC, std::size_t const StrideC,
                     std::map<std::string, std::string> const &keywords)
{

    auto nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
    auto nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;
    auto nsizeS = StrideS == 1 ? nSizeS : nSizeS / StrideS;
    auto nsizeC = StrideC == 1 ? nSizeC : nSizeC / StrideC;

    if (nsizeX != nsizeY)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
        return false;
    }
    if (nsizeS > 1)
    {
        if (nsizeX != nsizeS)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }
    if (nsizeC > 1)
    {
        if (nsizeX != nsizeC)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x, nSizeX, StrideX);
        PyObject *PyArrayY = PyArray<T>(y, nSizeY, StrideY);
        PyObject *PyArrayS = nsizeS > 1 ? PyArray<T>(s, nSizeS, StrideS) : PyFloat_FromDouble(static_cast<double>(s[0]));
        PyObject *PyArrayC = nsizeC > 1 ? PyArray<T>(c, nSizeC, StrideC) : PyFloat_FromDouble(static_cast<double>(c[0]));

        PyTuple_SetItem(args, 0, PyArrayX);
        PyTuple_SetItem(args, 1, PyArrayY);
        PyTuple_SetItem(args, 2, PyArrayS);
        PyTuple_SetItem(args, 3, PyArrayC);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // No patch boundary will be drawn
        // For non-filled markers, the edgecolors kwarg is ignored and forced to ‘face’ internally.
        PyDict_SetItemString(kwargs, "edgecolor", PyString_FromString("None"));
    }
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyscatter, args, kwargs);

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
    std::cerr << " Call to scatter failed!" << std::endl;
    return false;
}

/*!
 * \brief A scatter plot of y vs x with scaler marker size and color
 * 
 * \tparam T        Data type 
 *  
 * \param x         Scalar or array-like, data positions
 * \param nSizeX    Size of array x
 * \param StrideX   Stride element stride 
 * \param y         Scalar or array-like, data positions
 * \param nSizeY    Size of array y
 * \param StrideY   Stride element stride 
 * \param s         Scalar marker size in points**2
 * \param c         Scalar data colors
 * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::scatter(T const *x, int const nSizeX, std::size_t const StrideX,
                     T const *y, int const nSizeY, std::size_t const StrideY,
                     T const s, T const c,
                     std::map<std::string, std::string> const &keywords)
{

    auto nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
    auto nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;
    if (nsizeX != nsizeY)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
        return false;
    }

    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x, nSizeX, StrideX);
        PyObject *PyArrayY = PyArray<T>(y, nSizeY, StrideY);
        PyObject *PyArrayS = PyFloat_FromDouble(static_cast<double>(s));
        PyObject *PyArrayC = PyFloat_FromDouble(static_cast<double>(c));

        PyTuple_SetItem(args, 0, PyArrayX);
        PyTuple_SetItem(args, 1, PyArrayY);
        PyTuple_SetItem(args, 2, PyArrayS);
        PyTuple_SetItem(args, 3, PyArrayC);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // No patch boundary will be drawn
        // For non-filled markers, the edgecolors kwarg is ignored and forced to ‘face’ internally.
        PyDict_SetItemString(kwargs, "edgecolor", PyString_FromString("None"));
    }
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyscatter, args, kwargs);

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
    std::cerr << " Call to scatter failed!" << std::endl;
    return false;
}

/*!
 * \brief A scatter plot of y vs x with varying marker size and/or color
 * 
 * \tparam T        Data type 
 *  
 * \param x         Scalar or array-like, data positions
 * \param y         Scalar or array-like, data positions
 * \param s         Scalar or array-like, marker size in points**2
 * \param c         Scalar or array-like, data colors
 * \param nSize     Size of arrays
 * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::scatter(T const *x, T const *y, int const nSize,
                     T const *s, T const *c,
                     std::map<std::string, std::string> const &keywords)
{
    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x, nSize);
        PyObject *PyArrayY = PyArray<T>(y, nSize);
        PyObject *PyArrayS = PyArray<T>(s, nSize);
        PyObject *PyArrayC = PyArray<T>(c, nSize);

        PyTuple_SetItem(args, 0, PyArrayX);
        PyTuple_SetItem(args, 1, PyArrayY);
        PyTuple_SetItem(args, 2, PyArrayS);
        PyTuple_SetItem(args, 3, PyArrayC);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // No patch boundary will be drawn
        // For non-filled markers, the edgecolors kwarg is ignored and forced to ‘face’ internally.
        PyDict_SetItemString(kwargs, "edgecolor", PyString_FromString("None"));
    }
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyscatter, args, kwargs);

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
    std::cerr << " Call to scatter failed!" << std::endl;
    return false;
}

/*!
 * \brief A scatter plot of y vs x with scaler marker size and/or color
 * 
 * \tparam T        Data type 
 *  
 * \param x         Scalar or array-like, data positions
 * \param y         Scalar or array-like, data positions
 * \param nSize     Size of arrays
 * \param s         Scalar marker size in points**2
 * \param c         Scalar data color
 * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::scatter(T const *x, T const *y, int const nSize,
                     T const s, T const c,
                     std::map<std::string, std::string> const &keywords)
{
    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x, nSize);
        PyObject *PyArrayY = PyArray<T>(y, nSize);
        PyObject *PyArrayS = PyFloat_FromDouble(static_cast<double>(s));
        PyObject *PyArrayC = PyFloat_FromDouble(static_cast<double>(c));

        PyTuple_SetItem(args, 0, PyArrayX);
        PyTuple_SetItem(args, 1, PyArrayY);
        PyTuple_SetItem(args, 2, PyArrayS);
        PyTuple_SetItem(args, 3, PyArrayC);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // No patch boundary will be drawn
        // For non-filled markers, the edgecolors kwarg is ignored and forced to ‘face’ internally.
        PyDict_SetItemString(kwargs, "edgecolor", PyString_FromString("None"));
    }
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pyscatter, args, kwargs);

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
    std::cerr << " Call to scatter failed!" << std::endl;
    return false;
}

/*!
 * \brief Make a plot with log scaling on the x axis
 * This is just a thin wrapper around plot which additionally changes the x-axis to log scaling. 
 * All of the concepts and parameters of plot can be used here as well.
 * 
 * \tparam T     Data type 
 * 
 * \param x      Scalar or array-like, data positions
 * \param y      Scalar or array-like, data positions
 * \param fmt    Plot format string
 * \param label  object (Set the label to s for auto legend)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::semilogx(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt, std::string const &label)
{
    if (x.size() != y.size())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
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

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pysemilogx, args, kwargs);

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
 * \brief Make a plot with log scaling on the x axis
 * This is just a thin wrapper around plot which additionally changes the x-axis to log scaling. 
 * All of the concepts and parameters of plot can be used here as well.
 * 
 * \tparam T     Data type 
 * 
 * \param x         Scalar or array-like, data positions
 * \param nSizeX    Size of array x
 * \param StrideX   Stride element stride 
 * \param y         Scalar or array-like, data positions
 * \param nSizeY    Size of array y
 * \param StrideY   Stride element stride 
 * \param fmt    Plot format string
 * \param label  object (Set the label to s for auto legend)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::semilogx(T const *x, int const nSizeX, std::size_t const StrideX,
                      T const *y, int const nSizeY, std::size_t const StrideY,
                      std::string const &fmt, std::string const &label)
{
    {
        std::size_t nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
        std::size_t nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;

        if (nsizeX != nsizeY)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSizeX, StrideX);
        PyObject *yarray = PyArray<T>(y, nSizeY, StrideY);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pysemilogx, args, kwargs);

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
 * \brief Make a plot with log scaling on the x axis
 * This is just a thin wrapper around plot which additionally changes the x-axis to log scaling. 
 * All of the concepts and parameters of plot can be used here as well.
 * 
 * \tparam T     Data type 
 * 
 * \param x      Scalar or array-like, data positions
 * \param y      Scalar or array-like, data positions
 * \param nSize  Size of arrays
 * \param fmt    Plot format string
 * \param label  object (Set the label to s for auto legend)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::semilogx(T const *x, T const *y, int const nSize, std::string const &fmt, std::string const &label)
{
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSize);
        PyObject *yarray = PyArray<T>(y, nSize);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pysemilogx, args, kwargs);

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
 * \brief Make a plot with log scaling on the y axis
 * This is just a thin wrapper around plot which additionally changes the y-axis to log scaling. 
 * All of the concepts and parameters of plot can be used here as well.
 * 
 * \tparam T     Data type 
 * 
 * \param x      Scalar or array-like, data positions
 * \param y      Scalar or array-like, data positions
 * \param fmt    Plot format string
 * \param label  object (Set the label to s for auto legend)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::semilogy(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt, std::string const &label)
{
    if (x.size() != y.size())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
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

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pysemilogy, args, kwargs);

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
 * \brief Make a plot with log scaling on the y axis
 * This is just a thin wrapper around plot which additionally changes the y-axis to log scaling. 
 * All of the concepts and parameters of plot can be used here as well.
 * 
 * \tparam T     Data type 
 * 
 * \param x         Scalar or array-like, data positions
 * \param nSizeX    Size of array x
 * \param StrideX   Stride element stride 
 * \param y         Scalar or array-like, data positions
 * \param nSizeY    Size of array y
 * \param StrideY   Stride element stride 
 * \param fmt    Plot format string
 * \param label  object (Set the label to s for auto legend)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::semilogy(T const *x, int const nSizeX, std::size_t const StrideX,
                      T const *y, int const nSizeY, std::size_t const StrideY,
                      std::string const &fmt, std::string const &label)
{
    {
        std::size_t nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
        std::size_t nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;

        if (nsizeX != nsizeY)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSizeX, StrideX);
        PyObject *yarray = PyArray<T>(y, nSizeY, StrideY);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pysemilogy, args, kwargs);

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
 * \brief Make a plot with log scaling on the y axis
 * This is just a thin wrapper around plot which additionally changes the y-axis to log scaling. 
 * All of the concepts and parameters of plot can be used here as well.
 * 
 * \tparam T     Data type 
 * 
 * \param x      Scalar or array-like, data positions
 * \param y      Scalar or array-like, data positions
 * \param nSize  Size of arrays
 * \param fmt    Plot format string
 * \param label  object (Set the label to s for auto legend)
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::semilogy(T const *x, T const *y, int const nSize, std::string const &fmt, std::string const &label)
{
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSize);
        PyObject *yarray = PyArray<T>(y, nSize);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pysemilogy, args, kwargs);

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
bool pyplot::show(bool const block)
{
    PyObject *res;
    if (block)
    {
        res = PyObject_CallObject(pyplot::mpl.pyshow, pyplot::mpl.pyEmpty);

        if (res)
        {
            Py_DECREF(res);
            return true;
        }
    }
    else
    {
        // Create a new empty dictionary
        PyObject *kwargs = PyDict_New();
        // Construct keyword args
        PyDict_SetItemString(kwargs, "block", Py_False);

        res = PyObject_Call(pyplot::mpl.pyshow, pyplot::mpl.pyEmpty, kwargs);

        if (res)
        {
            Py_DECREF(res);
            Py_DECREF(kwargs);
            return true;
        }
        Py_DECREF(kwargs);
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
        std::cerr << "Two vectors should have the same size!" << std::endl;
        return false;
    }

    // Construct positional args
    PyObject *args = PyTuple_New(2);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *yarray = PyArray<T>(y);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pystem, args, kwargs);

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
 * \brief Create a stem plot.
 * A stem plot plots vertical lines at each x location from the baseline to y, and places a marker there
 * 
 * \tparam T        Data type 
 * 
 * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1)
 * \param nSizeX    Size of array x
 * \param StrideX   Stride element stride 
 * \param y         The y-values of the stem heads
 * \param nSizeY    Size of array y
 * \param StrideY   Stride element stride   
 * \param keywords  
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::stem(T const *x, int const nSizeX, std::size_t const StrideX,
                  T const *y, int const nSizeY, std::size_t const StrideY,
                  std::map<std::string, std::string> const &keywords)
{
    {
        std::size_t nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
        std::size_t nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;

        if (nsizeX != nsizeY)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    // Construct positional args
    PyObject *args = PyTuple_New(2);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSizeX, StrideX);
        PyObject *yarray = PyArray<T>(y, nSizeY, StrideY);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pystem, args, kwargs);

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
 * \brief Create a stem plot.
 * A stem plot plots vertical lines at each x location from the baseline to y, and places a marker there
 * 
 * \tparam T        Data type 
 * 
 * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1)
 * \param y         The y-values of the stem heads
 * \param nSize     Size of arrays   
 * \param keywords  
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::stem(T const *x, T const *y, int const nSize, std::map<std::string, std::string> const &keywords)
{
    // Construct positional args
    PyObject *args = PyTuple_New(2);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSize);
        PyObject *yarray = PyArray<T>(y, nSize);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pystem, args, kwargs);

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
 * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1).
 * \param y         The y-values of the stem heads.
 * \param fmt       A format string
 * \param label     object. Set the label to s for auto legend
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::stem(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt, std::string const &label)
{
    if (x.size() != y.size())
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << "Two vectors should have the same size!" << std::endl;
        return false;
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *yarray = PyArray<T>(y);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pystem, args, kwargs);

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
 * \tparam T        Data type 
 * 
 * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1)
 * \param nSizeX    Size of array x
 * \param StrideX   Stride element stride 
 * \param y         The y-values of the stem heads
 * \param nSizeY    Size of array y
 * \param StrideY   Stride element stride   
 * \param fmt       A format string
 * \param label     object. Set the label to s for auto legend
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::stem(T const *x, int const nSizeX, std::size_t const StrideX,
                  T const *y, int const nSizeY, std::size_t const StrideY,
                  std::string const &fmt, std::string const &label)
{
    {
        std::size_t nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
        std::size_t nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;

        if (nsizeX != nsizeY)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << "Two vectors should have the same size!" << std::endl;
            return false;
        }
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSizeX, StrideX);
        PyObject *yarray = PyArray<T>(y, nSizeY, StrideY);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pystem, args, kwargs);

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
 * \tparam T        Data type 
 * 
 * \param x         The x-positions of the stems. Default: (0, 1, …, len(y) - 1)
 * \param y         The y-values of the stem heads
 * \param nSize     Size of arrays  
 * \param fmt       A format string
 * \param label     object. Set the label to s for auto legend
 * 
 * \return true 
 * \return false 
 */
template <typename T>
bool pyplot::stem(T const *x, T const *y, int const nSize, std::string const &fmt, std::string const &label)
{
    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x, nSize);
        PyObject *yarray = PyArray<T>(y, nSize);
        PyObject *pystring = PyString_FromString(fmt.c_str());

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, pystring);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pystem, args, kwargs);

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
 * \brief Return a subplot axes at the given grid position
 * In the current figure, create and return an Axes, 
 * at position index of a (virtual) grid of nrows by ncols axes. 
 * Indexes go from 1 to nrows * ncols, incrementing in row-major order.
 * 
 * \param nrows 
 * \param ncols 
 * \param index 
 */
bool pyplot::subplot(long const nrows, long const ncols, long const index)
{
    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        PyTuple_SetItem(args, 0, PyFloat_FromDouble(nrows));
        PyTuple_SetItem(args, 1, PyFloat_FromDouble(ncols));
        PyTuple_SetItem(args, 2, PyFloat_FromDouble(index));
    }

    PyObject *res = PyObject_CallObject(pyplot::mpl.pysubplot, args);

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

    PyObject *res = PyObject_CallObject(pyplot::mpl.pytitle, args);

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
    PyObject *res = PyObject_CallObject(pyplot::mpl.pytight_layout, pyplot::mpl.pyEmpty);

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
        PyList_SetItem(list, 0, PyFloat_FromDouble(static_cast<double>(left)));
        PyList_SetItem(list, 1, PyFloat_FromDouble(static_cast<double>(right)));

        PyTuple_SetItem(args, 0, list);
    }

    PyObject *res = PyObject_CallObject(pyplot::mpl.pyxlim, args);

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
    PyObject *args = pyplot::mpl.pyEmpty;
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyxlim, args);
    PyObject *pleft = PyTuple_GetItem(res, 0);
    PyObject *pright = PyTuple_GetItem(res, 1);

    *left = static_cast<T>(PyFloat_AsDouble(pleft));
    *right = static_cast<T>(PyFloat_AsDouble(pright));

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

    PyObject *res = PyObject_CallObject(pyplot::mpl.pyxlabel, args);
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
    PyObject *res = PyObject_Call(pyplot::mpl.pyxkcd, pyplot::mpl.pyEmpty, kwargs);
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
        PyList_SetItem(list, 0, PyFloat_FromDouble(static_cast<double>(left)));
        PyList_SetItem(list, 1, PyFloat_FromDouble(static_cast<double>(right)));

        PyTuple_SetItem(args, 0, list);
    }

    PyObject *res = PyObject_CallObject(pyplot::mpl.pyylim, args);

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
    PyObject *args = pyplot::mpl.pyEmpty;
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyylim, args);
    PyObject *pleft = PyTuple_GetItem(res, 0);
    PyObject *pright = PyTuple_GetItem(res, 1);

    *left = static_cast<T>(PyFloat_AsDouble(pleft));
    *right = static_cast<T>(PyFloat_AsDouble(pright));

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

    PyObject *res = PyObject_CallObject(pyplot::mpl.pyylabel, args);
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

/*!
 * \brief Construct a new pyplot::matplotlib interpreter object
 * 
 */
pyplot::matplotlib::matplotlib()
{
// optional but recommended
#if PY_MAJOR_VERSION >= 3
    wchar_t name[] = L"umuq";
#else
    char name[] = "umuq";
#endif

    // Pass name to the Python interpreter
    Py_SetProgramName(name);

    // Initialize the Python interpreter. Required.
    Py_Initialize();

    // Initialize numpy
    import_array();

    PyObject *matplotlibModule = NULL;
    PyObject *pyplotModule = NULL;
    PyObject *pylabModule = NULL;

    {
        PyObject *matplotlibName = PyString_FromString("matplotlib");
        if (!matplotlibName)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Couldn't create matplotlib PyObject!" << std::endl;
            throw std::runtime_error("Error creating matplotlib PyObject!");
        }

        matplotlibModule = PyImport_Import(matplotlibName);
        if (!matplotlibModule)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Couldn't load matplotlib!" << std::endl;
            throw std::runtime_error("Error loading matplotlib!");
        }

        // Decrementing of the reference count
        Py_DECREF(matplotlibName);
    }

    {
        PyObject *pyplotName = PyString_FromString("matplotlib.pyplot");
        if (!pyplotName)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Couldn't create pyplot PyObject!" << std::endl;
            throw std::runtime_error("Error creating pyplot PyObject!");
        }

        pyplotModule = PyImport_Import(pyplotName);
        if (!pyplotModule)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Couldn't load matplotlib.pyplot!" << std::endl;
            throw std::runtime_error("Error loading module matplotlib.pyplot!");
        }

        // Decrementing of the reference count
        Py_DECREF(pyplotName);
    }

    // matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    // or matplotlib.backends is imported for the first time
    if (!backend.empty())
    {
        // Call the method named use of object matplotlib with a variable number of C arguments.
        PyObject_CallMethod(matplotlibModule, const_cast<char *>("use"), const_cast<char *>("s"), backend.c_str());
    }

    {
        PyObject *pylabName = PyString_FromString("pylab");
        if (!pylabName)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Couldn't create pylab PyObject!" << std::endl;
            throw std::runtime_error("Error creating pylab PyObject!");
        }

        pylabModule = PyImport_Import(pylabName);
        if (!pylabModule)
        {
            std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
            std::cerr << " Couldn't load pylab!" << std::endl;
            throw std::runtime_error("Error loading pylab!");
        }

        // Decrementing of the reference count
        Py_DECREF(pylabName);
    }

    // Retrieve an attribute named annotate from object pyplotModule.
    pyannotate = PyObject_GetAttrString(pyplotModule, "annotate");
    if (!pyannotate)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find annotate function!" << std::endl;
        throw std::runtime_error("Couldn't find annotate function!");
    }
    if (!PyFunction_Check(pyannotate))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named show from object pyplotModule.
    pyshow = PyObject_GetAttrString(pyplotModule, "show");
    if (!pyshow)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find show function!" << std::endl;
        throw std::runtime_error("Couldn't find show function!");
    }
    //Return true if it is a function object
    if (!PyFunction_Check(pyshow))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named close from object pyplotModule.
    pyclose = PyObject_GetAttrString(pyplotModule, "close");
    if (!pyclose)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find close function!" << std::endl;
        throw std::runtime_error("Couldn't find close function!");
    }
    if (!PyFunction_Check(pyclose))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named draw from object pyplotModule.
    pydraw = PyObject_GetAttrString(pyplotModule, "draw");
    if (!pydraw)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find draw function!" << std::endl;
        throw std::runtime_error("Couldn't find draw function!");
    }
    if (!PyFunction_Check(pydraw))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named pause from object pyplotModule.
    pypause = PyObject_GetAttrString(pyplotModule, "pause");
    if (!pypause)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find pause function!" << std::endl;
        throw std::runtime_error("Couldn't find pause function!");
    }
    if (!PyFunction_Check(pypause))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named figure from object pyplotModule.
    pyfigure = PyObject_GetAttrString(pyplotModule, "figure");
    if (!pyfigure)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find figure function!" << std::endl;
        throw std::runtime_error("Couldn't find figure function!");
    }
    if (!PyFunction_Check(pyfigure))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named plot from object pyplotModule.
    pyplot = PyObject_GetAttrString(pyplotModule, "plot");
    if (!pyplot)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find plot function!" << std::endl;
        throw std::runtime_error("Couldn't find plot function!");
    }
    if (!PyFunction_Check(pyplot))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named semilogx from object pyplotModule.
    pysemilogx = PyObject_GetAttrString(pyplotModule, "semilogx");
    if (!pysemilogx)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find semilogx function!" << std::endl;
        throw std::runtime_error("Couldn't find semilogx function!");
    }
    if (!PyFunction_Check(pysemilogx))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named semilogy from object pyplotModule.
    pysemilogy = PyObject_GetAttrString(pyplotModule, "semilogy");
    if (!pysemilogy)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find semilogy function!" << std::endl;
        throw std::runtime_error("Couldn't find semilogy function!");
    }
    if (!PyFunction_Check(pysemilogy))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named loglog from object pyplotModule.
    pyloglog = PyObject_GetAttrString(pyplotModule, "loglog");
    if (!pyloglog)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find loglog function!" << std::endl;
        throw std::runtime_error("Couldn't find loglog function!");
    }
    if (!PyFunction_Check(pyloglog))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named fill_between from object pyplotModule.
    pyfill_between = PyObject_GetAttrString(pyplotModule, "fill_between");
    if (!pyfill_between)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find fill_between function!" << std::endl;
        throw std::runtime_error("Couldn't find fill_between function!");
    }
    if (!PyFunction_Check(pyfill_between))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named hist from object pyplotModule.
    pyhist = PyObject_GetAttrString(pyplotModule, "hist");
    if (!pyhist)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find hist function!" << std::endl;
        throw std::runtime_error("Couldn't find hist function!");
    }
    if (!PyFunction_Check(pyhist))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named subplot from object pyplotModule.
    pysubplot = PyObject_GetAttrString(pyplotModule, "subplot");
    if (!pysubplot)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find subplot function!" << std::endl;
        throw std::runtime_error("Couldn't find subplot function!");
    }
    if (!PyFunction_Check(pysubplot))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named legend from object pyplotModule.
    pylegend = PyObject_GetAttrString(pyplotModule, "legend");
    if (!pylegend)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find legend function!" << std::endl;
        throw std::runtime_error("Couldn't find legend function!");
    }
    if (!PyFunction_Check(pylegend))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named ylim from object pyplotModule.
    pyylim = PyObject_GetAttrString(pyplotModule, "ylim");
    if (!pyylim)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find ylim function!" << std::endl;
        throw std::runtime_error("Couldn't find ylim function!");
    }
    if (!PyFunction_Check(pyylim))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named title from object pyplotModule.
    pytitle = PyObject_GetAttrString(pyplotModule, "title");
    if (!pytitle)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find title function!" << std::endl;
        throw std::runtime_error("Couldn't find title function!");
    }
    if (!PyFunction_Check(pytitle))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named axis from object pyplotModule.
    pyaxis = PyObject_GetAttrString(pyplotModule, "axis");
    if (!pyaxis)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find axis function!" << std::endl;
        throw std::runtime_error("Couldn't find axis function!");
    }
    if (!PyFunction_Check(pyaxis))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named xlabel from object pyplotModule.
    pyxlabel = PyObject_GetAttrString(pyplotModule, "xlabel");
    if (!pyxlabel)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find xlabel function!" << std::endl;
        throw std::runtime_error("Couldn't find xlabel function!");
    }
    if (!PyFunction_Check(pyxlabel))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named ylabel from object pyplotModule.
    pyylabel = PyObject_GetAttrString(pyplotModule, "ylabel");
    if (!pyylabel)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find ylabel function!" << std::endl;
        throw std::runtime_error("Couldn't find ylabel function!");
    }
    if (!PyFunction_Check(pyylabel))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named grid from object pyplotModule.
    pygrid = PyObject_GetAttrString(pyplotModule, "grid");
    if (!pygrid)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find grid function!" << std::endl;
        throw std::runtime_error("Couldn't find grid function!");
    }
    if (!PyFunction_Check(pygrid))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named xlim from object pyplotModule.
    pyxlim = PyObject_GetAttrString(pyplotModule, "xlim");
    if (!pyxlim)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find xlim function!" << std::endl;
        throw std::runtime_error("Couldn't find xlim function!");
    }
    if (!PyFunction_Check(pyxlim))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named ion from object pyplotModule.
    pyion = PyObject_GetAttrString(pyplotModule, "ion");
    if (!pyion)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find ion function!" << std::endl;
        throw std::runtime_error("Couldn't find ion function!");
    }
    if (!PyFunction_Check(pyion))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named savefig from object pylabModule.
    pysavefig = PyObject_GetAttrString(pylabModule, "savefig");
    if (!pysavefig)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find savefig function!" << std::endl;
        throw std::runtime_error("Couldn't find savefig function!");
    }
    if (!PyFunction_Check(pysavefig))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named scatter from object pyplotModule.
    pyscatter = PyObject_GetAttrString(pyplotModule, "scatter");
    if (!pyscatter)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find scatter function!" << std::endl;
        throw std::runtime_error("Couldn't find scatter function!");
    }
    if (!PyFunction_Check(pyscatter))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named cla from object pyplotModule.
    pycla = PyObject_GetAttrString(pyplotModule, "cla");
    if (!pycla)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find cla function!" << std::endl;
        throw std::runtime_error("Couldn't find cla function!");
    }
    if (!PyFunction_Check(pycla))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named clf from object pyplotModule.
    pyclf = PyObject_GetAttrString(pyplotModule, "clf");
    if (!pyclf)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find clf function!" << std::endl;
        throw std::runtime_error("Couldn't find clf function!");
    }
    if (!PyFunction_Check(pyclf))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named errorbar from object pyplotModule.
    pyerrorbar = PyObject_GetAttrString(pyplotModule, "errorbar");
    if (!pyerrorbar)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find errorbar function!" << std::endl;
        throw std::runtime_error("Couldn't find errorbar function!");
    }
    if (!PyFunction_Check(pyerrorbar))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named tight_layout from object pyplotModule.
    pytight_layout = PyObject_GetAttrString(pyplotModule, "tight_layout");
    if (!pytight_layout)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find tight_layout function!" << std::endl;
        throw std::runtime_error("Couldn't find tight_layout function!");
    }
    if (!PyFunction_Check(pytight_layout))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named stem from object pyplotModule.
    pystem = PyObject_GetAttrString(pyplotModule, "stem");
    if (!pystem)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find stem function!" << std::endl;
        throw std::runtime_error("Couldn't find stem function!");
    }
    if (!PyFunction_Check(pystem))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }
    // Retrieve an attribute named xkcd from object pyplotModule.
    pyxkcd = PyObject_GetAttrString(pyplotModule, "xkcd");
    if (!pyxkcd)
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Couldn't find xkcd function!" << std::endl;
        throw std::runtime_error("Couldn't find xkcd function!");
    }
    if (!PyFunction_Check(pyxkcd))
    {
        std::cerr << "Error : " << __FILE__ << ":" << __LINE__ << " : " << std::endl;
        std::cerr << " Python object is unexpectedly not a PyFunction !" << std::endl;
        throw std::runtime_error("Python object is unexpectedly not a PyFunction.");
    }

    //Return a new tuple object of size 0
    pyEmpty = PyTuple_New(0);
}

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

template <typename TIn, typename TOut>
PyObject *PyArray(std::vector<TIn> const &idata)
{
    PyObject *pArray;
    {
        npy_intp nsize = static_cast<npy_intp>(idata.size());
        if (NPIDatatype<TOut> != NPIDatatype<TIn>)
        {
            if (NPIDatatype<TOut> == NPY_NOTYPE)
            {
                std::vector<double> vd(nsize);
                std::copy(idata.begin(), idata.end(), vd.begin());
                pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_DOUBLE, (void *)(vd.data()));
            }
            else
            {
                std::vector<TOut> vd(nsize);
                std::copy(idata.begin(), idata.end(), vd.begin());
                pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<TOut>, (void *)(vd.data()));
            }
        }
        else
        {
            pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<TIn>, (void *)(idata.data()));
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
PyObject *PyArray(T *idata, int const nSize, std::size_t const Stride)
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

template <typename TIn, typename TOut>
PyObject *PyArray(TIn *idata, int const nSize, std::size_t const Stride)
{
    PyObject *pArray;
    {
        npy_intp nsize;

        if (Stride != 1)
        {
            ArrayWrapper<TIn> iArray(idata, nSize, Stride);
            nsize = static_cast<npy_intp>(iArray.size());
            if (NPIDatatype<TOut> != NPIDatatype<TIn>)
            {
                if (NPIDatatype<TOut> == NPY_NOTYPE)
                {
                    std::vector<double> vd(nsize);
                    std::copy(iArray.begin(), iArray.end(), vd.begin());
                    pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_DOUBLE, (void *)(vd.data()));
                    return pArray;
                }
            }
            std::vector<TOut> vd(nsize);
            std::copy(iArray.begin(), iArray.end(), vd.begin());
            pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<TOut>, (void *)(vd.data()));
            return pArray;
        }

        nsize = static_cast<npy_intp>(nSize);
        if (NPIDatatype<TOut> != NPIDatatype<TIn>)
        {
            if (NPIDatatype<TOut> == NPY_NOTYPE)
            {
                std::vector<double> vd(nsize);
                std::copy(idata, idata + nSize, vd.begin());
                pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_DOUBLE, (void *)(vd.data()));
            }
            else
            {
                std::vector<TOut> vd(nsize);
                std::copy(idata, idata + nSize, vd.begin());
                pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<TOut>, (void *)(vd.data()));
            }
        }
        else
        {
            pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<TIn>, (void *)(idata));
        }
    }
    return pArray;
}

#endif //HAVE_PYTHON
#endif //UMUQ_MATPLOTLIB_H
