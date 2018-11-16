#ifndef UMUQ_CONSTANTS_H
#define UMUQ_CONSTANTS_H

namespace umuq
{

/*! \defgroup Const_Module Constant module
 * \ingroup Core_Module
 *
 * This is the costant module of %UMUQ providing constant variable definitions.
 * Constant values: \f$ \pi,~2\pi,~\sqrt{\pi},~\sqrt{2\pi},~ln(\pi),~\text{and}~ln(2\pi)~\f$ are defined in 
 * this module.
 * 
 * Reference:<br>
 * http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html 
 */

#ifdef M_PI
#undef M_PI
#endif
#ifdef M_2PI
#undef M_2PI
#endif
#ifdef M_SPI
#undef M_SPI
#endif
#ifdef M_S2PI
#undef M_S2PI
#endif
#ifdef M_LPI
#undef M_LPI
#endif
#ifdef M_L2PI
#undef M_L2PI
#endif
#ifdef LINESIZE
#undef LINESIZE
#endif

/*!
 * \ingroup Const_Module
 * 
 * \brief \f$ \pi \f$
 */
#define M_PI 3.14159265358979323846264338327950288419716939937510582097494459230781640l

/*!
 * \ingroup Const_Module
 * 
 * \brief \f$ 2\pi \f$
 */
#define M_2PI 6.28318530717958647692528676655900576839433879875021164194988918461563281l

/*!
 * \ingroup Const_Module
 * 
 * \brief \f$ \sqrt{\pi}  \f$
 */
#define M_SPI 1.77245385090551602729816748334114518279754945612238712821380778985291128l

/*!
 * \ingroup Const_Module
 * 
 * \brief \f$ \sqrt{2\pi} \f$
 */
#define M_S2PI 2.50662827463100050241576528481104525300698674060993831662992357634229365l

/*!
 * \ingroup Const_Module
 * 
 * \brief \f$ \log{\pi} \f$
 */
#define M_LPI 1.14472988584940017414342735135305871164729481291531157151362307147213774l

/*!
 * \ingroup Const_Module
 * 
 * \brief \f$ \log{2\pi} \f$
 */
#define M_L2PI 1.83787706640934548356065947281123527972279494727556682563430308096553139l

/*!
 * \ingroup Const_Module
 * 
 * \brief Maximum size of a char * in %UMUQ
 */
#define LINESIZE 256

/*!
 * \ingroup Const_Module
 * 
 * \brief This value means that the evaluate an expression failed
 */
#define UFAIL -1e12

/*!
 * \brief This value means that the cost to evaluate an expression coefficient 
 * is either very expensive or cannot be known at compile time.
 * 
 */
int const HugeCost = 10000;

/*!
 * \ingroup Const_Module
 * 
 * \brief Get the machine precision accuracy for T data type
 * 
 * \tparam T Data type
 */
template <typename T>
static T const machinePrecision = std::pow(T{10}, -digits10<T>());

/*!
 * \ingroup Const_Module
 * 
 * \brief Empty vector for initialization
 * 
 * \tparam T Data type
 */
template <typename T>
static std::vector<T> const EmptyVector{};

/*!
 * \ingroup Const_Module
 * 
 * \brief Empty string for initialization
 * 
 */
static std::string const EmptyString{};

/*!
 * \brief Manual (or static) loop unrolling increment 
 * 
 * \todo
 * Later, it should be updated to use the compiler optimization
 */
#define unrolledIncrement 4

#if unrolledIncrement == 0 || unrolledIncrement == 4 || unrolledIncrement == 6 || unrolledIncrement == 8 || unrolledIncrement == 10 || unrolledIncrement == 12
#else
#undef unrolledIncrement
#define unrolledIncrement 0
#endif

} // namespace umuq

#endif // UMUQ_CONSTANTS
