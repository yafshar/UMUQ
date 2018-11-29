#ifndef UMUQ_PRIORTYPE_H
#define UMUQ_PRIORTYPE_H

namespace umuq
{

/*! \enum priorTypes
 * \ingroup Numerics_Module 
 * 
 * \brief Prior distribution types currently supported in %UMUQ
 * 
 */
enum class priorTypes
{
	/*! \link umuq::density::uniformDistribution UNIFORM \endlink */
	UNIFORM = 0,
	/*! \link umuq::density::gaussianDistribution GAUSSIAN \endlink */
	GAUSSIAN = 1,
	/*! \link umuq::density::exponentialDistribution EXPONENTIAL \endlink */
	EXPONENTIAL = 2,
	/*! \link umuq::density::gammaDistribution GAMMA \endlink */
	GAMMA = 3,
	/*! COMPOSITE  */
	COMPOSITE = 4
};

} // namespace umuq

#endif // UMUQ_PRIORTYPE
