#ifndef UMUQ_PYPLOT_H
#define UMUQ_PYPLOT_H
#ifdef HAVE_PYTHON

namespace umuq
{

inline namespace matplotlib_223
{

/*!
 * \file io/pyplot.hpp
 * \brief This module contains functions that allows to generate many kinds of plots
 *
 * The pyplot Module contains additions, adaptations and modifications to the 
 * original c++ interface to the matplotlib source codes made available under 
 * the following LICENSE:
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
 * Matplotlib library made available under the following LICENSE:
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

#include "../data/npydatatype.hpp"

} // namespace matplotlib_223
} // namespace umuq

#include "../misc/arraywrapper.hpp"

namespace umuq
{

/*! \namespace umuq::matplotlib_223
 * \ingroup IO_Module
 * 
 * \brief It contains several common approaches to plotting with Matplotlib python 2D library
 * 
 * It contains several common approaches to plotting with Matplotlib python 2D library from Matplotlib version 2.2.3
 * 
 */
inline namespace matplotlib_223
{

/*! \fn PyArray
 * \ingroup IO_Module
 * 
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

/*! \fn PyArray
 * \ingroup IO_Module
 * 
 * \brief Converts a data idata to Python array of size nSize
 * 
 * \tparam T Data type
 * 
 * \param idata  Input data
 * \param nSize  Size of the requested array
 * 
 * \return PyObject* Python array
 */
template <typename T>
PyObject *PyArray(T const idata, int const nSize);

template <typename TIn, typename TOut>
PyObject *PyArray(TIn const idata, int const nSize);

/*! \fn PyArray
 * \ingroup IO_Module
 * 
 * \brief Converts a data array idata to Python array
 * 
 * \tparam T Data type
 * 
 * \param idata   Input array of data
 * \param nSize   Size of the array
 * \param Stride  Element stride (default is 1)
 * 
 * \return PyObject* Python array
 */
template <typename T>
PyObject *PyArray(T const *idata, int const nSize, std::size_t const Stride = 1);

template <typename TIn, typename TOut>
PyObject *PyArray(TIn const *idata, int const nSize, std::size_t const Stride = 1);

/*! \fn Py2DArray
 * \ingroup IO_Module
 * 
 * \brief Converts a data array idata to the Python 2D array 
 * 
 * \tparam T Data type 
 * 
 * \param idata  Input array of data (with size of nDimX*nDimY)
 * \param nDimX  X size in the 2D array
 * \param nDimY  Y size in the 2D array
 * 
 * \returns PyObject* Python 2D array
 */
template <typename T>
PyObject *Py2DArray(std::vector<T> const &idata, int const nDimX, int const nDimY);

template <typename TIn, typename TOut>
PyObject *Py2DArray(std::vector<TIn> const &idata, int const nDimX, int const nDimY);

template <typename T>
PyObject *Py2DArray(T const *idata, int const nDimX, int const nDimY);

/*! \var static std::string backend
 * \ingroup IO_Module
 * 
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

/*! \fn setbackend
 * \ingroup IO_Module
 * 
 * \brief Set the “backend” to any of user interface backends
 * 
 * \param WXbackends user interface backends (for use in pygtk, wxpython, tkinter, qt4, or macosx; 
 *                   also referred to as “interactive backends”) or hardcopy backends to make image 
 *                   files (PNG, SVG, PDF, PS; also referred to as “non-interactive backends”)
 * 
 * \note 
 * - Must be called before the first regular call to matplotlib to have any effect
 * 
 * \note 
 * - Backend name specifications are not case-sensitive; e.g., ‘GTKAgg’ and ‘gtkagg’ are equivalent. 
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
 * GTKAgg       Agg rendering to a GTK 2.x canvas (requires PyGTK and pycairo or cairocffi; Python2 only)
 * GTK3Agg      Agg rendering to a GTK 3.x canvas (requires PyGObject and pycairo or cairocffi)
 * GTK          GDK rendering to a GTK 2.x canvas (not recommended and d eprecated in 2.0) (requires PyGTK and pycairo or cairocffi; Python2 only)
 * GTKCairo     Cairo rendering to a GTK 2.x canvas (requires PyGTK and pycairo or cairocffi; Python2 only)
 * GTK3Cairo    Cairo rendering to a GTK 3.x canvas (requires PyGObject and pycairo or cairocffi)
 * WXAgg        Agg rendering to to a wxWidgets canvas (requires wxPython)
 * WX           Native wxWidgets drawing to a wxWidgets Canvas (not recommended and deprecated in 2.0) (requires wxPython)
 * TkAgg        Agg rendering to a Tk canvas (requires TkInter)
 * macosx       Cocoa rendering in OSX windows (presently lacks blocking show() behavior when matplotlib is in non-interactive mode)
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

/*! \var constexpr char const DefaultColors
 * \ingroup IO_Module
 * 
 * \brief The following color abbreviations are supported
 * 
 * <table>
 * <caption id="multi_row">Colors</caption>
 * <tr><th> Character <th> Color        
 * <tr><td> 'b'       <td> blue  
 * <tr><td> 'g'       <td> green    
 * <tr><td> 'r'       <td> red     
 * <tr><td> 'c'       <td> cyan 
 * <tr><td> 'm'       <td> magenta
 * <tr><td> 'y'       <td> yellow
 * <tr><td> 'k'       <td> black
 * <tr><td> 'w'       <td> white
 * </table>
 */
constexpr char const DefaultColors[] = {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'};

/*! \var int const DefaultColorsSize
 * \ingroup IO_Module
 * 
 * \brief Size of the color abbreviations maps
 * 
 */
int const DefaultColorsSize = 8;

// colors = {u'c': (0.0, 0.75, 0.75), u'b': (0.0, 0.0, 1.0), u'w': (1.0, 1.0, 1.0), u'g': (0.0, 0.5, 0.0), u'y': (0.75, 0.75, 0), u'k': (0.0, 0.0, 0.0), u'r': (1.0, 0.0, 0.0), u'm': (0.75, 0, 0.75)}

/*! \class pyplot
 * \ingroup IO_Module
 * 
 * \brief This module contains several common approaches to plotting with Matplotlib python 2D library
 *
 * It contains below functions that allow you to generate many kinds of plots quickly:
 * 
 * - \b annotate      Annotate the point xy with text s
 * - \b axis          Convenience method to get or set axis properties
 * - \b cla           Clear the current axis
 * - \b clf           Clear the current figure
 * - \b close         Close a figure window
 * - \b contour       Plot contours
 * - \b contourf      Plot filled contours
 * - \b draw          Redraw the current figure
 * - \b errorbar      Plot y versus x as lines and/or markers with attached errorbars
 * - \b figure        Creates a new figure
 * - \b fill_between  Fill the area between two horizontal curves
 * - \b grid          Turn the axes grids on or off
 * - \b hist          Plot a histogram
 * - \b ion           Turn interactive mode on
 * - \b legend        Places a legend on the axes
 * - \b loglog        Make a plot with log scaling on both the x and y axis
 * - \b pause         Pause for interval seconds
 * - \b plot          Plot y versus x as lines and/or markers
 * - \b savefig       Save the current figure
 * - \b scatter       A scatter plot of y vs x with varying marker size and/or color
 * - \b semilogx      Make a plot with log scaling on the x axis
 * - \b semilogy      Make a plot with log scaling on the y axis
 * - \b show          Display a figure
 * - \b stem          Create a stem plot
 * - \b subplot       Return a subplot axes at the given grid position
 * - \b title         Set a title of the current axes
 * - \b tight_layout  Automatically adjust subplot parameters to give specified padding
 * - \b xlim          Set/Get the x limits of the current axes
 * - \b xlabel        Set the x-axis label of the current axes
 * - \b xkcd          Turns on xkcd sketch-style drawing mode
 * - \b ylim          Set/Get the y limits of the current axes
 * - \b ylabel        Set the y-axis label of the current axes
 * 
 * 
 * The following format string characters are accepted to control the line style or marker:
 * <table>
 * <caption id="multi_row">Character to control the line style or marker</caption>
 * <tr><th> Character <th> Description        
 * <tr><td> '-'       <td> solid line style
 * <tr><td> '--'      <td> dashed line style
 * <tr><td> '-.'      <td> dash-dot line style
 * <tr><td> ':'       <td> dotted line style
 * <tr><td> '.'       <td> point marker
 * <tr><td> ','       <td> pixel marker
 * <tr><td> 'o'       <td> circle marker
 * <tr><td> 'v'       <td> triangle_down marker
 * <tr><td> '^'       <td> triangle_up marker
 * <tr><td> '<'       <td> triangle_left marker
 * <tr><td> '>'       <td> triangle_right marker
 * <tr><td> '1'       <td> tri_down marker
 * <tr><td> '2'       <td> tri_up marker
 * <tr><td> '3'       <td> tri_left marker
 * <tr><td> '4'       <td> tri_right marker
 * <tr><td> 's'       <td> square marker
 * <tr><td> 'p'       <td> pentagon marker
 * <tr><td> '*'       <td> star marker
 * <tr><td> 'h'       <td> hexagon1 marker
 * <tr><td> 'H'       <td> hexagon2 marker
 * <tr><td> '+'       <td> plus marker
 * <tr><td> 'x'       <td> x marker
 * <tr><td> 'D'       <td> diamond marker
 * <tr><td> 'd'       <td> thin_diamond marker
 * <tr><td> '|'       <td> vline marker
 * <tr><td> '_'       <td> hline marker
 * </table>
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
     * \brief Get the name of the current backend
     * 
     * \return The name of the current backend
     */
    inline std::string get_backend();

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
     * \return false If it encounters an unexpected problem
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
     * \brief Draw contour lines
     * 
     * \tparam T Data type
     * 
     * \param x         Array-like (1D arrays representing the x coordinates of a grid)
     *                  such that len(x) is the number of columns in z
     * \param y         Array-like (1D arrays representing the y coordinates of a grid)
     *                  such that len(y) is the number of rows in z
     * \param z         Array-like with size [len(x) * len(y)] grids. 
     *                  The height values over which the contour is drawn.
     * \param levels    Int scalar value or array-like, optional
     *                  If array-like, draw contour lines at the specified levels. 
     *                  The values must be in increasing order.
     * \param keywords  All other keyword arguments are passed on to contour. They control it's properties.
     * 
     * \returns true 
     * \returns false 
     */
    template <typename T>
    inline bool contour(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z, std::vector<T> const &levels,
                        std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    template <typename T>
    inline bool contour(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z, int const levels,
                        std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    template <typename T>
    inline bool contour(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z,
                        std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief Draw filled contour
     *
     * \tparam T Data type
     *
     * \param x         Array-like (1D arrays representing the x coordinates of a grid)
     *                  such that len(x) is the number of columns in z
     * \param y         Array-like (1D arrays representing the y coordinates of a grid)
     *                  such that len(y) is the number of rows in z
     * \param z         Array-like with size [len(x) * len(y)] grids. 
     *                  The height values over which the contour is drawn.
     * \param levels    Int scalar value or array-like, optional
     *                  If array-like, draw contour lines at the specified levels. 
     *                  The values must be in increasing order.
     * \param keywords  All other keyword arguments are passed on to contour. They control it's properties.
     *
     * \returns true
     * \returns false
     */
    template <typename T>
    inline bool contourf(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z, std::vector<T> const &levels,
                         std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    template <typename T>
    inline bool contourf(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z, int const levels,
                         std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    template <typename T>
    inline bool contourf(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z,
                         std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     *                  The marker color. Possible values:
     *                      - A single color format, \sa DefaultColors.
     *                      - A sequence of color specifications of length n equals to length of array x.
     *                      - A sequence of n numbers to be mapped to colors using cmap and norm.
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    template <typename T>
    bool scatter(std::vector<T> const &x, std::vector<T> const &y,
                 std::vector<int> const &s, std::vector<T> const &c,
                 std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    template <typename T>
    bool scatter(std::vector<T> const &x, std::vector<T> const &y,
                 std::vector<int> const &s, std::string const c = "k",
                 std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    /*!
     * \brief A scatter plot of y vs x with scaler marker size and color
     * 
     * \tparam T Data type 
     * 
     * \param x         Scalar or array-like, data positions
     * \param y         Scalar or array-like, data positions
     * \param s         Scalar marker size in points**2
     * \param c         The marker color. Possible value:
     *                      - A single color format, \sa DefaultColors.
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    template <typename T>
    bool scatter(std::vector<T> const &x, std::vector<T> const &y,
                 int const s, std::string const c = "k",
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
     *                  The marker color. Possible values:
     *                      - A single color format, \sa DefaultColors.
     *                      - A sequence of color specifications of length n equals to length of array x.
     *                      - A sequence of n numbers to be mapped to colors using cmap and norm.
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    template <typename T>
    bool scatter(T const *x, int const nSizeX, std::size_t const StrideX,
                 T const *y, int const nSizeY, std::size_t const StrideY,
                 int const *s, int const nSizeS, std::size_t const StrideS,
                 T const *c, int const nSizeC, std::size_t const StrideC,
                 std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    template <typename T>
    bool scatter(T const *x, int const nSizeX, std::size_t const StrideX,
                 T const *y, int const nSizeY, std::size_t const StrideY,
                 int const *s, int const nSizeS, std::size_t const StrideS,
                 std::string const c = "k",
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
     * \param c         The marker color. Possible value:
     *                      - A single color format, \sa DefaultColors.
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    template <typename T>
    bool scatter(T const *x, int const nSizeX, std::size_t const StrideX,
                 T const *y, int const nSizeY, std::size_t const StrideY,
                 int const s, std::string const c = "k",
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
     *                  The marker color. Possible values:
     *                      - A single color format, \sa DefaultColors.
     *                      - A sequence of color specifications of length n equals to length of array x.
     *                      - A sequence of n numbers to be mapped to colors using cmap and norm.
     * \param nSize     Size of arrays
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    template <typename T>
    bool scatter(T const *x, T const *y, int const nSize,
                 int const *s, T const *c,
                 std::map<std::string, std::string> const &keywords = std::map<std::string, std::string>());

    template <typename T>
    bool scatter(T const *x, T const *y, int const nSize,
                 int const *s, std::string const c = "k",
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
     * \param c         The marker color. Possible value:
     *                      - A single color format, \sa DefaultColors.
     * \param keywords  keywords are used to specify properties like a line label (for auto legends), linewidth, antialiasing, marker face color.
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    template <typename T>
    bool scatter(T const *x, T const *y, int const nSize,
                 int const s, std::string const c = "k",
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
     * \return false If it encounters an unexpected problem
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
     * \param fmt       Plot format string
     * \param label     Object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
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
     * \param label  Object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
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
     * \param label  Object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    template <typename T>
    bool semilogy(std::vector<T> const &x, std::vector<T> const &y,
                  std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Make a plot with log scaling on the y axis
     * This is just a thin wrapper around plot which additionally changes the y-axis to log scaling. 
     * All of the concepts and parameters of plot can be used here as well.
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
     * \param label     Object (Set the label to s for auto legend)
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
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
     * \return false If it encounters an unexpected problem
     */
    template <typename T>
    bool stem(T const *x, T const *y, int const nSize,
              std::string const &fmt = "", std::string const &label = "");

    /*!
     * \brief Return a subplot axes at the given grid position
     * In the current figure, create and return an Axes, at position index of a (virtual) grid of nDimX by nDimY axes. 
     * Indexes go from 1 to nDimX * nDimY, incrementing in row-major order.
     * 
     * \param nDimX 
     * \param nDimY 
     * \param index 
     */
    bool subplot(long const nDimX, long const nDimY, long const index);

    /*!
     * \brief Set a title of the current axes
     * 
     * \param label Text to use for the title
     */
    bool title(std::string const &label);

    /*!
     * \brief Automatically adjust subplot parameters to give specified padding
     * 
     * \todo
     * We should call this automatically for every plot!
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
    /*!
     * \brief Make construct noncopyable
     * 
     */
    pyplot(pyplot const &) = delete;

    /*!
     * \brief Make it not assignable
     * 
     * \returns pyplot& 
     */
    pyplot &operator=(pyplot const &) = delete;

  private:
    /*! \class matplotlib
     * \ingroup IO_Module
     * 
     * \brief This class sets and initializes python matplotlib for different use cases
     * 
     * \verbatim
     * Matplotlib is a Python 2D plotting library which produces publication quality figures 
     * in a variety of hardcopy formats and interactive environments across platforms. 
     * 
     * To support all of use cases, matplotlib can target different outputs, and each of these 
     * capabilities is called a backend the “frontend” is the user facing code, i.e., the 
     * plotting code, whereas the “backend” does all the hard work behind-the-scenes to make 
     * the figure.
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
         * \brief Construct a new matplotlib object
         * 
         */
        matplotlib();

        /*!
         * \brief Destroy the matplotlib object
         * 
         */
        ~matplotlib()
        {
            // Undo all initializations made by Py_Initialize() and subsequent use
            // of Python/C API functions
            Py_Finalize();
        }

      public:
        // Make it noncopyable
        matplotlib(matplotlib const &) = delete;

        // Make it not assignable
        matplotlib &operator=(matplotlib const &) = delete;

      public:
        /*!
         * \brief Backend object
         * 
         */
        PyObject *pyget_backend;

        /*!
         * \brief Tuple object
         * 
         */
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
        //! Plot contours
        PyObject *pycontour;
        //! Plot filled contours
        PyObject *pycontourf;
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

pyplot::matplotlib pyplot::mpl;

inline std::string pyplot::get_backend()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyget_backend, pyplot::mpl.pyEmpty);
    if (res)
    {
        std::string backendName = PyString_AsString(res);
        Py_DECREF(res);
        return backendName;
    }
    UMUQFAILRETURNSTRING("Couldn't get the name of the current backend!");
}

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
    UMUQFAILRETURN("Call to annotate failed!");
}

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
    UMUQFAILRETURN("Call to axis failed!");
}

inline bool pyplot::cla()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pycla, pyplot::mpl.pyEmpty);

    if (res)
    {
        Py_DECREF(res);
        return true;
    }
    UMUQFAILRETURN("Call to cla failed!");
}

inline bool pyplot::clf()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyclf, pyplot::mpl.pyEmpty);

    if (res)
    {
        Py_DECREF(res);
        return true;
    }
    UMUQFAILRETURN("Call to clf failed!");
}

inline bool pyplot::close()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyclose, pyplot::mpl.pyEmpty);

    if (res)
    {
        Py_DECREF(res);
        return true;
    }
    UMUQFAILRETURN("Call to close failed!");
}

template <typename T>
inline bool pyplot::contour(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z, std::vector<T> const &levels,
                            std::map<std::string, std::string> const &keywords)
{

    if (x.size() * y.size() != z.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the correct size! (len(z) != len(x) * len(y))");
    }

    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        int const nDimX = static_cast<int>(x.size());
        int const nDimY = static_cast<int>(y.size());

        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *yarray = PyArray<T>(y);
        PyObject *zarray = Py2DArray<T>(z, nDimX, nDimY);
        PyObject *larray = PyArray<T>(levels);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, zarray);
        PyTuple_SetItem(args, 3, larray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pycontour, args, kwargs);

    if (res)
    {
        Py_DECREF(res);
        Py_DECREF(kwargs);
        Py_DECREF(args);
        return true;
    }
    Py_DECREF(kwargs);
    Py_DECREF(args);
    UMUQFAILRETURN("Call to contour failed!");
}

template <typename T>
inline bool pyplot::contour(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z, int const levels,
                            std::map<std::string, std::string> const &keywords)
{

    if (x.size() * y.size() != z.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the correct size! (len(z) != len(x) * len(y))");
    }

    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        int const nDimX = static_cast<int>(x.size());
        int const nDimY = static_cast<int>(y.size());

        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *yarray = PyArray<T>(y);
        PyObject *zarray = Py2DArray<T>(z, nDimX, nDimY);
        PyObject *larray = PyInt_FromLong(levels);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, zarray);
        PyTuple_SetItem(args, 3, larray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pycontour, args, kwargs);

    if (res)
    {
        Py_DECREF(res);
        Py_DECREF(kwargs);
        Py_DECREF(args);
        return true;
    }
    Py_DECREF(kwargs);
    Py_DECREF(args);
    UMUQFAILRETURN("Call to contour failed!");
}

template <typename T>
inline bool pyplot::contour(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z,
                            std::map<std::string, std::string> const &keywords)
{

    if (x.size() * y.size() != z.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the correct size! (len(z) != len(x) * len(y))");
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        int const nDimX = static_cast<int>(x.size());
        int const nDimY = static_cast<int>(y.size());

        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *yarray = PyArray<T>(y);
        PyObject *zarray = Py2DArray<T>(z, nDimX, nDimY);

        std::cout << std::endl;
        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, zarray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pycontour, args, kwargs);

    if (res)
    {
        Py_DECREF(res);
        Py_DECREF(kwargs);
        Py_DECREF(args);
        return true;
    }
    Py_DECREF(kwargs);
    Py_DECREF(args);
    UMUQFAILRETURN("Call to contour failed!");
}

template <typename T>
inline bool pyplot::contourf(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z, std::vector<T> const &levels,
                             std::map<std::string, std::string> const &keywords)
{

    if (x.size() * y.size() != z.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the correct size! (len(z) != len(x) * len(y))");
    }

    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        int const nDimX = static_cast<int>(x.size());
        int const nDimY = static_cast<int>(y.size());

        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *yarray = PyArray<T>(y);
        PyObject *zarray = Py2DArray<T>(z, nDimX, nDimY);
        PyObject *larray = PyArray<T>(levels);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, zarray);
        PyTuple_SetItem(args, 3, larray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pycontourf, args, kwargs);

    if (res)
    {
        Py_DECREF(res);
        Py_DECREF(kwargs);
        Py_DECREF(args);
        return true;
    }
    Py_DECREF(kwargs);
    Py_DECREF(args);
    UMUQFAILRETURN("Call to contourf failed!");
}

template <typename T>
inline bool pyplot::contourf(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z, int const levels,
                             std::map<std::string, std::string> const &keywords)
{

    if (x.size() * y.size() != z.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the correct size! (len(z) != len(x) * len(y))");
    }

    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        int const nDimX = static_cast<int>(x.size());
        int const nDimY = static_cast<int>(y.size());

        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *yarray = PyArray<T>(y);
        PyObject *zarray = Py2DArray<T>(z, nDimX, nDimY);
        PyObject *larray = PyInt_FromLong(levels);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, zarray);
        PyTuple_SetItem(args, 3, larray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pycontourf, args, kwargs);

    if (res)
    {
        Py_DECREF(res);
        Py_DECREF(kwargs);
        Py_DECREF(args);
        return true;
    }
    Py_DECREF(kwargs);
    Py_DECREF(args);
    UMUQFAILRETURN("Call to contourf failed!");
}

template <typename T>
inline bool pyplot::contourf(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &z,
                             std::map<std::string, std::string> const &keywords)
{

    if (x.size() * y.size() != z.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the correct size! (len(z) != len(x) * len(y))");
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        int const nDimX = static_cast<int>(x.size());
        int const nDimY = static_cast<int>(y.size());

        // Using numpy arrays
        PyObject *xarray = PyArray<T>(x);
        PyObject *yarray = PyArray<T>(y);
        PyObject *zarray = Py2DArray<T>(z, nDimX, nDimY);

        PyTuple_SetItem(args, 0, xarray);
        PyTuple_SetItem(args, 1, yarray);
        PyTuple_SetItem(args, 2, zarray);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        // Construct keyword args
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject *res = PyObject_Call(pyplot::mpl.pycontourf, args, kwargs);

    if (res)
    {
        Py_DECREF(res);
        Py_DECREF(kwargs);
        Py_DECREF(args);
        return true;
    }
    Py_DECREF(kwargs);
    Py_DECREF(args);
    UMUQFAILRETURN("Call to contourf failed!");
}

inline bool pyplot::draw()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pydraw, pyplot::mpl.pyEmpty);

    if (res)
    {
        Py_DECREF(res);
        return true;
    }
    UMUQFAILRETURN("Call to draw failed!");
}

template <typename T>
bool pyplot::errorbar(std::vector<T> const &x, std::vector<T> const &y, std::vector<T> const &yerr, std::string const &fmt)
{
    if (x.size() != y.size() || x.size() != yerr.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to errorbar failed!");
}

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
            UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to errorbar failed!");
}

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
    UMUQFAILRETURN("Call to errorbar failed!");
}

inline bool pyplot::figure()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyfigure, pyplot::mpl.pyEmpty);

    if (res)
    {
        Py_DECREF(res);
        return true;
    }
    UMUQFAILRETURN("Call to figure failed!");
}

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
    UMUQFAILRETURN("Call to figure failed!");
}

template <typename T>
bool pyplot::fill_between(std::vector<T> const &x, std::vector<T> const &y1, std::vector<T> const &y2, std::map<std::string, std::string> const &keywords)
{
    if (x.size() != y1.size() || x.size() != y2.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to fill_between failed!");
}

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
            UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to fill_between failed!");
}

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
    UMUQFAILRETURN("Call to fill_between failed!");
}

bool pyplot::grid(bool flag)
{
    PyObject *args = PyTuple_New(1);
    {
        PyTuple_SetItem(args, 0, flag ? Py_True : Py_False);
    }

    PyObject *res = PyObject_CallObject(pyplot::mpl.pygrid, args);

    if (res)
    {
        Py_DECREF(res);
        Py_DECREF(args);
        return true;
    }
    Py_DECREF(args);
    UMUQFAILRETURN("Call to grid failed!");
}

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
            /*!
             * \note 
             * - In some cases density keyword does not work and one has to use normed instead
             */
            PyDict_SetItemString(kwargs, "density", Py_True);
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
    UMUQFAILRETURN("Call to hist failed!");
}

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
            PyDict_SetItemString(kwargs, "density", Py_True);
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
    UMUQFAILRETURN("Call to hist failed!");
}

inline bool pyplot::ion()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pyion, pyplot::mpl.pyEmpty);

    if (res)
    {
        Py_DECREF(res);
        return true;
    }
    UMUQFAILRETURN("Call to ion failed!");
}

inline bool pyplot::legend()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pylegend, pyplot::mpl.pyEmpty);

    if (res)
    {
        Py_DECREF(res);
        return true;
    }
    UMUQFAILRETURN("Call to legend failed!");
}

template <typename T>
bool pyplot::loglog(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt, std::string const &label)
{
    if (x.size() != y.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to loglog failed!");
}

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
            UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to loglog failed!");
}

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
    UMUQFAILRETURN("Call to loglog failed!");
}

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
    UMUQFAILRETURN("Call to pause failed!");
}

template <typename T>
bool pyplot::plot(std::vector<T> const &x, std::vector<T> const &y, std::map<std::string, std::string> const &keywords)
{
    if (x.size() != y.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to plot failed!");
}

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
            UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to plot failed!");
}

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
    UMUQFAILRETURN("Call to plot failed!");
}

template <typename T>
bool pyplot::plot(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt, std::string const &label)
{
    if (x.size() != y.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to plot failed!");
}

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
            UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to plot failed!");
}

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
    UMUQFAILRETURN("Call to plot failed!");
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
    UMUQFAILRETURN("Call to savefig failed!");
}

template <typename T>
bool pyplot::scatter(std::vector<T> const &x, std::vector<T> const &y,
                     std::vector<int> const &s, std::vector<T> const &c,
                     std::map<std::string, std::string> const &keywords)
{
    if (x.size() != y.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
    }
    if (s.size() > 1)
    {
        if (x.size() != s.size())
        {
            UMUQFAILRETURN("Two input vectors do not have the same size!");
        }
    }
    if (x.size() != c.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
    }

    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x);
        PyObject *PyArrayY = PyArray<T>(y);
        PyObject *PyArrayS = s.size() > 1 ? PyArray<int>(s) : PyInt_FromLong(s[0]);
        PyObject *PyArrayC = PyArray<T>(c);

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
    UMUQFAILRETURN("Call to scatter failed!");
}

template <typename T>
bool pyplot::scatter(std::vector<T> const &x, std::vector<T> const &y,
                     std::vector<int> const &s, std::string const c,
                     std::map<std::string, std::string> const &keywords)
{
    if (x.size() != y.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
    }
    if (s.size() > 1)
    {
        if (x.size() != s.size())
        {
            UMUQFAILRETURN("Two input vectors do not have the same size!");
        }
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x);
        PyObject *PyArrayY = PyArray<T>(y);
        PyObject *PyArrayS = s.size() > 1 ? PyArray<int>(s) : PyInt_FromLong(s[0]);

        PyTuple_SetItem(args, 0, PyArrayX);
        PyTuple_SetItem(args, 1, PyArrayY);
        PyTuple_SetItem(args, 2, PyArrayS);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        PyDict_SetItemString(kwargs, "c", PyString_FromString(c.c_str()));
    }
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
    UMUQFAILRETURN("Call to scatter failed!");
}

template <typename T>
bool pyplot::scatter(std::vector<T> const &x, std::vector<T> const &y,
                     int const s, std::string const c,
                     std::map<std::string, std::string> const &keywords)
{
    if (x.size() != y.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x);
        PyObject *PyArrayY = PyArray<T>(y);
        PyObject *PyArrayS = PyInt_FromLong(s);

        PyTuple_SetItem(args, 0, PyArrayX);
        PyTuple_SetItem(args, 1, PyArrayY);
        PyTuple_SetItem(args, 2, PyArrayS);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        PyDict_SetItemString(kwargs, "c", PyString_FromString(c.c_str()));
    }
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
    UMUQFAILRETURN("Call to scatter failed!");
}

template <typename T>
bool pyplot::scatter(T const *x, int const nSizeX, std::size_t const StrideX,
                     T const *y, int const nSizeY, std::size_t const StrideY,
                     int const *s, int const nSizeS, std::size_t const StrideS,
                     T const *c, int const nSizeC, std::size_t const StrideC,
                     std::map<std::string, std::string> const &keywords)
{

    auto nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
    auto nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;
    auto nsizeS = StrideS == 1 ? nSizeS : nSizeS / StrideS;
    auto nsizeC = StrideC == 1 ? nSizeC : nSizeC / StrideC;

    if (nsizeX != nsizeY)
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
    }
    if (nsizeS > 1)
    {
        if (nsizeX != nsizeS)
        {
            UMUQFAILRETURN("Two input vectors do not have the same size!");
        }
    }
    if (nsizeX != nsizeC)
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
    }

    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x, nSizeX, StrideX);
        PyObject *PyArrayY = PyArray<T>(y, nSizeY, StrideY);
        PyObject *PyArrayS = nsizeS > 1 ? PyArray<int>(s, nSizeS, StrideS) : PyInt_FromLong(s[0]);
        PyObject *PyArrayC = PyArray<T>(c, nsizeC, StrideC);

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
    UMUQFAILRETURN("Call to scatter failed!");
}

template <typename T>
bool pyplot::scatter(T const *x, int const nSizeX, std::size_t const StrideX,
                     T const *y, int const nSizeY, std::size_t const StrideY,
                     int const s, std::string const c,
                     std::map<std::string, std::string> const &keywords)
{

    auto nsizeX = StrideX == 1 ? nSizeX : nSizeX / StrideX;
    auto nsizeY = StrideY == 1 ? nSizeY : nSizeY / StrideY;
    if (nsizeX != nsizeY)
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
    }

    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x, nSizeX, StrideX);
        PyObject *PyArrayY = PyArray<T>(y, nSizeY, StrideY);
        PyObject *PyArrayS = PyInt_FromLong(s);

        PyTuple_SetItem(args, 0, PyArrayX);
        PyTuple_SetItem(args, 1, PyArrayY);
        PyTuple_SetItem(args, 2, PyArrayS);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        PyDict_SetItemString(kwargs, "c", PyString_FromString(c.c_str()));
    }
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
    UMUQFAILRETURN("Call to scatter failed!");
}

template <typename T>
bool pyplot::scatter(T const *x, T const *y, int const nSize,
                     int const *s, T const *c,
                     std::map<std::string, std::string> const &keywords)
{
    // Construct positional args
    PyObject *args = PyTuple_New(4);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x, nSize);
        PyObject *PyArrayY = PyArray<T>(y, nSize);
        PyObject *PyArrayS = PyArray<int>(s, nSize);
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
    UMUQFAILRETURN("Call to scatter failed!");
}

template <typename T>
bool pyplot::scatter(T const *x, T const *y, int const nSize,
                     int const s, std::string const c,
                     std::map<std::string, std::string> const &keywords)
{
    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        // Using numpy arrays
        PyObject *PyArrayX = PyArray<T>(x, nSize);
        PyObject *PyArrayY = PyArray<T>(y, nSize);
        PyObject *PyArrayS = PyInt_FromLong(s);

        PyTuple_SetItem(args, 0, PyArrayX);
        PyTuple_SetItem(args, 1, PyArrayY);
        PyTuple_SetItem(args, 2, PyArrayS);
    }

    // Create a new empty dictionary
    PyObject *kwargs = PyDict_New();
    {
        PyDict_SetItemString(kwargs, "c", PyString_FromString(c.c_str()));
    }
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
    UMUQFAILRETURN("Call to scatter failed!");
}

template <typename T>
bool pyplot::semilogx(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt, std::string const &label)
{
    if (x.size() != y.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to semilogx failed!");
}

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
            UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to semilogx failed!");
}

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
    UMUQFAILRETURN("Call to semilogx failed!");
}

template <typename T>
bool pyplot::semilogy(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt, std::string const &label)
{
    if (x.size() != y.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to semilogy failed!");
}

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
            UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to semilogy failed!");
}

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
    UMUQFAILRETURN("Call to semilogy failed!");
}

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
    UMUQFAILRETURN("Call to show failed!");
}

template <typename T>
bool pyplot::stem(std::vector<T> const &x, std::vector<T> const &y, std::map<std::string, std::string> const &keywords)
{
    if (x.size() != y.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to stem failed!");
}

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
            UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to stem failed!");
}

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
    UMUQFAILRETURN("Call to stem failed!");
}

template <typename T>
bool pyplot::stem(std::vector<T> const &x, std::vector<T> const &y, std::string const &fmt, std::string const &label)
{
    if (x.size() != y.size())
    {
        UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to stem failed!");
}

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
            UMUQFAILRETURN("Two input vectors do not have the same size!");
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
    UMUQFAILRETURN("Call to stem failed!");
}

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
    UMUQFAILRETURN("Call to stem failed!");
}

bool pyplot::subplot(long const nDimX, long const nDimY, long const index)
{
    // Construct positional args
    PyObject *args = PyTuple_New(3);
    {
        PyTuple_SetItem(args, 0, PyFloat_FromDouble(nDimX));
        PyTuple_SetItem(args, 1, PyFloat_FromDouble(nDimY));
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
    UMUQFAILRETURN("Call to subplot failed!");
}

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
    UMUQFAILRETURN("Call to title failed!");
}

inline bool pyplot::tight_layout()
{
    PyObject *res = PyObject_CallObject(pyplot::mpl.pytight_layout, pyplot::mpl.pyEmpty);

    if (res)
    {
        Py_DECREF(res);
        return true;
    }
    UMUQFAILRETURN("Call to tight_layout failed!");
}

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
    UMUQFAILRETURN("Call to xlim failed!");
}

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
    UMUQFAILRETURN("Call to xlim failed!");
}

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
    UMUQFAILRETURN("Call to xlabel failed!");
}

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
    UMUQFAILRETURN("Call to xkcd failed!");
}

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
    UMUQFAILRETURN("Call to ylim failed!");
}

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
    UMUQFAILRETURN("Call to ylim failed!");
}

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
    UMUQFAILRETURN("Call to ylabel failed!");
}

pyplot::matplotlib::matplotlib()
{
// optional but recommended
#if PY_MAJOR_VERSION >= 3
    wchar_t name[] = L"umuq";
#else
    char name[] = "umuq";
#endif

    // Pass name to the Python
    Py_SetProgramName(name);

    // Initialize the Python. Required.
    Py_Initialize();

    // Initialize numpy
    import_array();

    PyObject *matplotlibModule = NULL;
    PyObject *pyplotModule = NULL;
    PyObject *pylabModule = NULL;

    {
        // import matplotlib

        PyObject *matplotlibName = PyString_FromString("matplotlib");
        if (!matplotlibName)
        {
            UMUQFAIL("Error creating matplotlib PyObject!");
        }

        matplotlibModule = PyImport_Import(matplotlibName);
        if (!matplotlibModule)
        {
            UMUQFAIL("Error loading matplotlib!");
        }

        // Decrementing of the reference count
        Py_DECREF(matplotlibName);
    }

    {
        // import matplotlib.pyplot

        PyObject *pyplotName = PyString_FromString("matplotlib.pyplot");
        if (!pyplotName)
        {
            UMUQFAIL("Error creating matplotlib.pyplot PyObject!");
        }

        pyplotModule = PyImport_Import(pyplotName);
        if (!pyplotModule)
        {
            UMUQFAIL("Error loading matplotlib.pyplot!");
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
        // import pylab

        PyObject *pylabName = PyString_FromString("pylab");
        if (!pylabName)
        {
            UMUQFAIL("Error creating pylab PyObject!");
        }

        pylabModule = PyImport_Import(pylabName);
        if (!pylabModule)
        {
            UMUQFAIL("Error loading pylab!");
        }

        // Decrementing of the reference count
        Py_DECREF(pylabName);
    }

    // Retrieve an attribute named get_backend from object matplotlibModule.
    pyget_backend = PyObject_GetAttrString(matplotlibModule, "get_backend");
    if (!pyget_backend)
    {
        UMUQFAIL("Couldn't find get_backend function!");
    }
    if (!PyFunction_Check(pyget_backend))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }

    //Return a new tuple object of size 0
    pyEmpty = PyTuple_New(0);

    // Retrieve an attribute named annotate from object pyplotModule.
    pyannotate = PyObject_GetAttrString(pyplotModule, "annotate");
    if (!pyannotate)
    {
        UMUQFAIL("Couldn't find annotate function!");
    }
    if (!PyFunction_Check(pyannotate))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named axis from object pyplotModule.
    pyaxis = PyObject_GetAttrString(pyplotModule, "axis");
    if (!pyaxis)
    {
        UMUQFAIL("Couldn't find axis function!");
    }
    if (!PyFunction_Check(pyaxis))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named cla from object pyplotModule.
    pycla = PyObject_GetAttrString(pyplotModule, "cla");
    if (!pycla)
    {
        UMUQFAIL("Couldn't find cla function!");
    }
    if (!PyFunction_Check(pycla))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named clf from object pyplotModule.
    pyclf = PyObject_GetAttrString(pyplotModule, "clf");
    if (!pyclf)
    {
        UMUQFAIL("Couldn't find clf function!");
    }
    if (!PyFunction_Check(pyclf))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named close from object pyplotModule.
    pyclose = PyObject_GetAttrString(pyplotModule, "close");
    if (!pyclose)
    {
        UMUQFAIL("Couldn't find close function!");
    }
    if (!PyFunction_Check(pyclose))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named contour from object pyplotModule.
    pycontour = PyObject_GetAttrString(pyplotModule, "contour");
    if (!pycontour)
    {
        UMUQFAIL("Couldn't find contour function!");
    }
    if (!PyFunction_Check(pycontour))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named contourf from object pyplotModule.
    pycontourf = PyObject_GetAttrString(pyplotModule, "contourf");
    if (!pycontourf)
    {
        UMUQFAIL("Couldn't find contourf function!");
    }
    if (!PyFunction_Check(pycontourf))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named draw from object pyplotModule.
    pydraw = PyObject_GetAttrString(pyplotModule, "draw");
    if (!pydraw)
    {
        UMUQFAIL("Couldn't find draw function!");
    }
    if (!PyFunction_Check(pydraw))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named errorbar from object pyplotModule.
    pyerrorbar = PyObject_GetAttrString(pyplotModule, "errorbar");
    if (!pyerrorbar)
    {
        UMUQFAIL("Couldn't find errorbar function!");
    }
    if (!PyFunction_Check(pyerrorbar))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named figure from object pyplotModule.
    pyfigure = PyObject_GetAttrString(pyplotModule, "figure");
    if (!pyfigure)
    {
        UMUQFAIL("Couldn't find figure function!");
    }
    if (!PyFunction_Check(pyfigure))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named fill_between from object pyplotModule.
    pyfill_between = PyObject_GetAttrString(pyplotModule, "fill_between");
    if (!pyfill_between)
    {
        UMUQFAIL("Couldn't find fill_between function!");
    }
    if (!PyFunction_Check(pyfill_between))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named grid from object pyplotModule.
    pygrid = PyObject_GetAttrString(pyplotModule, "grid");
    if (!pygrid)
    {
        UMUQFAIL("Couldn't find grid function!");
    }
    if (!PyFunction_Check(pygrid))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named hist from object pyplotModule.
    pyhist = PyObject_GetAttrString(pyplotModule, "hist");
    if (!pyhist)
    {
        UMUQFAIL("Couldn't find hist function!");
    }
    if (!PyFunction_Check(pyhist))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named ion from object pyplotModule.
    pyion = PyObject_GetAttrString(pyplotModule, "ion");
    if (!pyion)
    {
        UMUQFAIL("Couldn't find ion function!");
    }
    if (!PyFunction_Check(pyion))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named legend from object pyplotModule.
    pylegend = PyObject_GetAttrString(pyplotModule, "legend");
    if (!pylegend)
    {
        UMUQFAIL("Couldn't find legend function!");
    }
    if (!PyFunction_Check(pylegend))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named loglog from object pyplotModule.
    pyloglog = PyObject_GetAttrString(pyplotModule, "loglog");
    if (!pyloglog)
    {
        UMUQFAIL("Couldn't find loglog function!");
    }
    if (!PyFunction_Check(pyloglog))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named pause from object pyplotModule.
    pypause = PyObject_GetAttrString(pyplotModule, "pause");
    if (!pypause)
    {
        UMUQFAIL("Couldn't find pause function!");
    }
    if (!PyFunction_Check(pypause))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named plot from object pyplotModule.
    pyplot = PyObject_GetAttrString(pyplotModule, "plot");
    if (!pyplot)
    {
        UMUQFAIL("Couldn't find plot function!");
    }
    if (!PyFunction_Check(pyplot))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named savefig from object pylabModule.
    pysavefig = PyObject_GetAttrString(pylabModule, "savefig");
    if (!pysavefig)
    {
        UMUQFAIL("Couldn't find savefig function!");
    }
    if (!PyFunction_Check(pysavefig))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named scatter from object pyplotModule.
    pyscatter = PyObject_GetAttrString(pyplotModule, "scatter");
    if (!pyscatter)
    {
        UMUQFAIL("Couldn't find scatter function!");
    }
    if (!PyFunction_Check(pyscatter))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named semilogx from object pyplotModule.
    pysemilogx = PyObject_GetAttrString(pyplotModule, "semilogx");
    if (!pysemilogx)
    {
        UMUQFAIL("Couldn't find semilogx function!");
    }
    if (!PyFunction_Check(pysemilogx))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named semilogy from object pyplotModule.
    pysemilogy = PyObject_GetAttrString(pyplotModule, "semilogy");
    if (!pysemilogy)
    {
        UMUQFAIL("Couldn't find semilogy function!");
    }
    if (!PyFunction_Check(pysemilogy))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named show from object pyplotModule.
    pyshow = PyObject_GetAttrString(pyplotModule, "show");
    if (!pyshow)
    {
        UMUQFAIL("Couldn't find show function!");
    }
    //Return true if it is a function object
    if (!PyFunction_Check(pyshow))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named stem from object pyplotModule.
    pystem = PyObject_GetAttrString(pyplotModule, "stem");
    if (!pystem)
    {
        UMUQFAIL("Couldn't find stem function!");
    }
    if (!PyFunction_Check(pystem))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named subplot from object pyplotModule.
    pysubplot = PyObject_GetAttrString(pyplotModule, "subplot");
    if (!pysubplot)
    {
        UMUQFAIL("Couldn't find subplot function!");
    }
    if (!PyFunction_Check(pysubplot))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named tight_layout from object pyplotModule.
    pytight_layout = PyObject_GetAttrString(pyplotModule, "tight_layout");
    if (!pytight_layout)
    {
        UMUQFAIL("Couldn't find tight_layout function!");
    }
    if (!PyFunction_Check(pytight_layout))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named title from object pyplotModule.
    pytitle = PyObject_GetAttrString(pyplotModule, "title");
    if (!pytitle)
    {
        UMUQFAIL("Couldn't find title function!");
    }
    if (!PyFunction_Check(pytitle))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named xlabel from object pyplotModule.
    pyxlabel = PyObject_GetAttrString(pyplotModule, "xlabel");
    if (!pyxlabel)
    {
        UMUQFAIL("Couldn't find xlabel function!");
    }
    if (!PyFunction_Check(pyxlabel))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named xlim from object pyplotModule.
    pyxlim = PyObject_GetAttrString(pyplotModule, "xlim");
    if (!pyxlim)
    {
        UMUQFAIL("Couldn't find xlim function!");
    }
    if (!PyFunction_Check(pyxlim))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named xkcd from object pyplotModule.
    pyxkcd = PyObject_GetAttrString(pyplotModule, "xkcd");
    if (!pyxkcd)
    {
        UMUQFAIL("Couldn't find xkcd function!");
    }
    if (!PyFunction_Check(pyxkcd))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named ylabel from object pyplotModule.
    pyylabel = PyObject_GetAttrString(pyplotModule, "ylabel");
    if (!pyylabel)
    {
        UMUQFAIL("Couldn't find ylabel function!");
    }
    if (!PyFunction_Check(pyylabel))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
    // Retrieve an attribute named ylim from object pyplotModule.
    pyylim = PyObject_GetAttrString(pyplotModule, "ylim");
    if (!pyylim)
    {
        UMUQFAIL("Couldn't find ylim function!");
    }
    if (!PyFunction_Check(pyylim))
    {
        UMUQFAIL("Python object unexpectedly is not a PyFunction!");
    }
}

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

template <typename T>
PyObject *PyArray(T const idata, int const nSize)
{
    PyObject *pArray;
    {
        npy_intp nsize = static_cast<npy_intp>(nSize);
        if (NPIDatatype<T> == NPY_NOTYPE)
        {
            std::vector<double> vd(nsize);
            std::fill(vd.begin(), vd.end(), static_cast<double>(idata));
            pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_DOUBLE, (void *)(vd.data()));
        }
        else
        {
            std::vector<T> vd(nsize);
            std::fill(vd.begin(), vd.end(), idata);
            pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<T>, (void *)(vd.data()));
        }
    }
    return pArray;
}

template <>
PyObject *PyArray<char>(char const idata, int const nSize)
{
    PyObject *pArray;
    {
        npy_intp nsize(1);
        std::string vd(nSize, idata);
        pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_STRING, (void *)(vd.c_str()));
    }
    return pArray;
}

template <typename TIn, typename TOut>
PyObject *PyArray(TIn const idata, int const nSize)
{
    PyObject *pArray;
    {
        npy_intp nsize = static_cast<npy_intp>(nSize);
        if (NPIDatatype<TOut> != NPIDatatype<TIn>)
        {
            if (NPIDatatype<TOut> == NPY_NOTYPE)
            {
                std::vector<double> vd(nsize);
                std::fill(vd.begin(), vd.end(), static_cast<double>(idata));
                pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_DOUBLE, (void *)(vd.data()));
            }
            else
            {
                std::vector<TOut> vd(nsize);
                std::fill(vd.begin(), vd.end(), static_cast<TOut>(idata));
                pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<TOut>, (void *)(vd.data()));
            }
        }
        else
        {
            std::vector<TIn> vd(nsize);
            std::fill(vd.begin(), vd.end(), idata);
            pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<TIn>, (void *)(idata.data()));
        }
    }
    return pArray;
}

template <typename T>
PyObject *PyArray(T const *idata, int const nSize, std::size_t const Stride)
{
    PyObject *pArray;
    {
        npy_intp nsize;

        if (Stride != 1)
        {
            arrayWrapper<T> iArray(idata, nSize, Stride);
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

// template <typename TIn, typename TOut>
// PyObject *PyArray(TIn const *idata, int const nSize, std::size_t const Stride)
// {
//     PyObject *pArray;
//     {
//         npy_intp nsize;

//         if (Stride != 1)
//         {
//             arrayWrapper<TIn> iArray(idata, nSize, Stride);
//             nsize = static_cast<npy_intp>(iArray.size());
//             if (NPIDatatype<TOut> != NPIDatatype<TIn>)
//             {
//                 if (NPIDatatype<TOut> == NPY_NOTYPE)
//                 {
//                     std::vector<double> vd(nsize);
//                     std::copy(iArray.begin(), iArray.end(), vd.begin());
//                     pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_DOUBLE, (void *)(vd.data()));
//                     return pArray;
//                 }
//             }
//             std::vector<TOut> vd(nsize);
//             std::copy(iArray.begin(), iArray.end(), vd.begin());
//             pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<TOut>, (void *)(vd.data()));
//             return pArray;
//         }

//         nsize = static_cast<npy_intp>(nSize);
//         if (NPIDatatype<TOut> != NPIDatatype<TIn>)
//         {
//             if (NPIDatatype<TOut> == NPY_NOTYPE)
//             {
//                 std::vector<double> vd(nsize);
//                 std::copy(idata, idata + nSize, vd.begin());
//                 pArray = PyArray_SimpleNewFromData(1, &nsize, NPY_DOUBLE, (void *)(vd.data()));
//             }
//             else
//             {
//                 std::vector<TOut> vd(nsize);
//                 std::copy(idata, idata + nSize, vd.begin());
//                 pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<TOut>, (void *)(vd.data()));
//             }
//         }
//         else
//         {
//             pArray = PyArray_SimpleNewFromData(1, &nsize, NPIDatatype<TIn>, (void *)(idata));
//         }
//     }
//     return pArray;
// }

template <typename T>
PyObject *Py2DArray(std::vector<T> const &idata, int const nDimX, int const nDimY)
{
    if (idata.size() != static_cast<decltype(idata.size())>(nDimX) * nDimY)
    {
        UMUQFAIL("Data size does not match with mesh numbers!");
    }

    PyObject *pArray;
    {
        npy_intp PyArrayDims[] = {nDimY, nDimX};
        if (NPIDatatype<T> == NPY_NOTYPE)
        {
            std::vector<double> vd(idata.size());
            std::copy(idata.begin(), idata.end(), vd.begin());
            pArray = PyArray_SimpleNewFromData(2, PyArrayDims, NPY_DOUBLE, (void *)(vd.data()));
        }
        else
        {
            pArray = PyArray_SimpleNewFromData(2, PyArrayDims, NPIDatatype<T>, (void *)(idata.data()));
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
    {
        npy_intp PyArrayDims[] = {nDimY, nDimX};
        if (NPIDatatype<TOut> != NPIDatatype<TIn>)
        {
            if (NPIDatatype<TOut> == NPY_NOTYPE)
            {
                std::vector<double> vd(idata.size());
                std::copy(idata.begin(), idata.end(), vd.begin());
                pArray = PyArray_SimpleNewFromData(2, PyArrayDims, NPY_DOUBLE, (void *)(vd.data()));
            }
            else
            {
                std::vector<TOut> vd(idata.size());
                std::copy(idata.begin(), idata.end(), vd.begin());
                pArray = PyArray_SimpleNewFromData(2, PyArrayDims, NPIDatatype<TOut>, (void *)(vd.data()));
            }
        }
        else
        {
            pArray = PyArray_SimpleNewFromData(2, PyArrayDims, NPIDatatype<TIn>, (void *)(idata.data()));
        }
    }
    return pArray;
}

template <typename T>
PyObject *Py2DArray(T const *idata, int const nDimX, int const nDimY)
{
    PyObject *pArray;
    {
        npy_intp PyArrayDims[] = {nDimY, nDimX};
        if (NPIDatatype<T> == NPY_NOTYPE)
        {
            std::vector<double> vd{idata, idata + nDimX * nDimY};
            pArray = PyArray_SimpleNewFromData(2, PyArrayDims, NPY_DOUBLE, (void *)(vd.data()));
        }
        else
        {
            pArray = PyArray_SimpleNewFromData(2, PyArrayDims, NPIDatatype<T>, (void *)(idata));
        }
    }
    return pArray;
}

} // namespace matplotlib_223
} // namespace umuq

#else

namespace umuq
{

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
            std::vector<double> vd{idata, idata + nRows * nCols};
     * 
     */
    ~pyplot() {}

  private:
    // Make it noncopyable
    pyplot(pyplot const &) = delete;

    // Make it not assignable
    pyplot &operator=(pyplot const &) = delete;
};

} // namespace umuq

#endif // HAVEPYTHON
#endif // UMUQ_PYPLOT
