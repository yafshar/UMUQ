#ifndef UMUQ_SURROGATEBASE_H
#define UMUQ_SURROGATEBASE_H

/*!
 * \brief Base class for different surrogate models
 * 
 * Basic interface for using surrogates. Surrogate maps an \f$ N-\text{dimensionals} \f$ 
 * parameter space to the real space. That is \f$ f: \mathcal{R}^N \rightarrow \mathcal{R} \f$.
 * The idea is that we have some surrogate of the parameter-to-data map so that
 * it can be used in the likelihood classes. Subclasses will define the 
 * particular surrogate model. Other classes will be used to build up the
 * surrogate from the user's model.
 */
template <typename T>
class surrogatebase
{
  public:
    /*!
     * \brief Construct a new surrogate Base object
     * 
     */
    surrogatebase(){};

    /*!
     * \brief Destroy the surrogate Base object
     * 
     */
    ~surrogatebase(){};

    /*!
     * \brief Method to return value given the parameter vector
     * 
     * \param  InputVector 
     * \return T 
     */
    T evaluate(T const *InputVector) const = 0;
};

#endif //UMUQ_SURROGATEBASE_H
