#ifndef UMUQ_PSRANDOM_H
#define UMUQ_PSRANDOM_H

#include "../../core/core.hpp"
#include "../factorial.hpp"
#include "saruprng.hpp"

//! RNG seed
static std::size_t PRNG_seed = 0;

//! 32-bit Mersenne Twister by Matsumoto and Nishimura, 1998
static std::vector<std::mt19937> NumberGenerator(1);

//! C++ Saru PRNG
static std::vector<Saru> saru(1);

//! True if PRNG state has been initialized, and false otherwise (logical).
static bool PRNG_initialized = false;

//! Muex object
static std::mutex PRNG_m;

/*! \class psrandom
  *
  * This class generates pseudo-random numbers.
  * Engines and distributions used to produce random values. 
  * 
  * All of the engines may be specifically seeded, for use with repeatable simulators.
  * Random number engines generate pseudo-random numbers using seed data as entropy source. 
  * The choice of which engine to use involves a number of tradeoffs: 
  * 
  * Saru PRNG has only a small storage requirement for state which is 64-bit and is very fast. 
  * 
  * The Mersenne twister is slower and has greater state storage requirements but with the right parameters has 
  * the longest non-repeating sequence with the most desirable spectral characteristics (for a given definition of desirable). 
  * 
  * \tparam T  Data type one of float or double
  */
template <typename T = double>
class psrandom
{
  public:
	/*!
     * \brief Construct a new psrandom object
     * 
     */
	psrandom();

	/*!
     * \brief Construct a new psrandom object
     * 
     * \param iseed Input seed for random number initialization 
     */
	explicit psrandom(int const iseed);

	/*!
     * \brief Destroy the psrandom object
     * 
     */
	~psrandom() {}

	/*!
     * \brief Init task on each node to set the current state of the engine for all the threads on that node 
     */
	static void init_Task();

	/*!
     * \brief Set the State of psrandom object
     * 
     * \return \a true when sets the current state of the engine successfully 
     */
	bool init();

	/*!
     * \brief Set the State of psrandom object
     * 
     * \return \a true when sets the current state of the engine successfully
     */
	bool setState();

	/*!
     * \brief Set the Seed object
     * 
     * \param iseed  Input seed for random number initialization 
     * 
     * \return true  when sets the seed of the engine successfully
     * \return false If the PRNG is already initialized or the state has been set
     */
	inline bool setSeed(int const iseed);

	/*!
     * \returns Uniform random number between \f$ [a \cdots b) \f$
     * 
     * \brief Uniform random number between \f$ [a \cdots b) \f$
     *
     * \tparam T data type one of float or double
     *
     * Advance the PRNG state by 1, and output a T precision \f$ [a \cdots b) \f$ number (default \f$ a = 0, b = 1 \f$)
     */
	inline T unirnd(T const a = 0, T const b = 1);

	/*!
     * \returns a uniform random number of a double precision \f$ [0 \cdots 1) \f$ floating point 
     * 
     * \brief Advance state by 1, and output a double precision \f$ [0 \cdots 1) \f$ floating point
     * 
     * Reference:
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, Comput. Phys. Comm. 184 (2013), 1119–1128.
     */
	inline double d();

	/*!
     * \returns a uniform random number of a single precision \f$ [0 \cdots 1) \f$ floating point
     * 
     * \Advance state by 1, and output a single precision \f$ [0 \cdots 1) \f$ floating point
     * 
     * Reference:
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, Comput. Phys. Comm. 184 (2013), 1119–1128.
     */
	inline float f();

	/*!
     * \returns an unisgned 32 bit integer pseudo-random value
     * 
     * \brief Advance state by 1, and output a 32 bit integer pseudo-random value.
     * 
     * Reference:
     * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, Comput. Phys. Comm. 184 (2013), 1119–1128.
     */
	inline unsigned int u32();

	/*! \fn multinomial
     * \brief The multinomial random distribution
     *  
     * \param    p        Vector of probabilities \f$ p_1, \cdots, p_k \f$
     * \param    K        Size of vector which shows K possible mutually exclusive outcomes 
     * \param    N        N independent trials
     * \param    mndist   A random sample from the multinomial distribution (with size of K)
     * 
     * 
     * NOTE: This should be called after setting the State of psrandom object
     * 
     * 
     * Let \f$ X=\left( X_1, \cdots, X_K \right) \f$ have a multinomial distribution \f$ M_K\left(N, p\right) \f$
     * The distribution of \f$ X \f$ is given by:
     * \f[
     *     Pr(X_1=n_1, \cdots, X_K=n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \cdots p_K^{n_K}
     * \f] 
     *
     * where \f$ n_1, \cdots n_K \f$ are nonnegative integers satisfying \f$ sum_{i=1}^{K} {n_i} = N\f$,
     * and \f$p = \left(p_1, \cdots, p_K\right)\f$ is a probability distribution. 
     *
     * Random variates are generated using the conditional binomial method.
     * This scales well with N and does not require a setup step.
     *   
     *  Ref: 
     *  C.S. David, The computer generation of multinomial random variates,
     *  Comp. Stat. Data Anal. 16 (1993) 205-217
     */
	bool multinomial(T const *p, unsigned int const K, unsigned int const N, unsigned int *mndist);

	/*! \fn Multinomial
     * \brief The multinomial distribution
     * 
     * NOTE: This can be called without setting the State of psrandom object
     */
	bool Multinomial(T const *p, unsigned int const K, unsigned int const N, unsigned int *mndist);

	/*!
     * \brief The Fisher-Yates shuffle is used to permute randomly given input array.
     *
	 * \tparam D Input data type
	 * 
     * \param idata array of input data of type int
     * \param nSize Size of the array idata
     *
     * 
     * NOTE: This should be called after setting the State of psrandom object
     * 
     * 
     * The permutations generated by this algorithm occur with the same probability.
     *
     * References : 
     * R. Durstenfeld, "Algorithm 235: Random permutation" Communications of the ACM, 7 (1964), p. 420
     */
	template <typename D>
	inline void shuffle(D *idata, int const nSize);

	/*!
     * \brief The Fisher-Yates shuffle is used to permute randomly given input array.
     *
	 * \tparam D Input data type
	 * 
     * \param idata array of input data of type int
     * \param nSize Size of the array idata
     * 
     * NOTE: This can be called without setting the State of psrandom object
     */
	template <typename D>
	inline void Shuffle(D *idata, int const nSize);

  private:
	// Make it noncopyable
	psrandom(psrandom const &) = delete;

	// Make it not assignable
	psrandom &operator=(psrandom const &) = delete;

  public:
	/*! \class normalDistribution
     * \brief Generates random numbers according to the Normal (or Gaussian) random number distribution
     * 
     * NOTE: This should be called after setting the State of psrandom object
     * 
     */
	class normalDistribution
	{
	  public:
		/*!
         * \brief Construct a new normalDistribution object (default mean = 0, stddev = 1)
         * 
         * \param mean    Mean
         * \param stddev  Standard deviation
         * 
         */
		normalDistribution(T mean = T{}, T stddev = T{1});

		/*!
         * \brief Move constructor, construct a new normalDistribution object from input normalDistribution object
         * 
         * \param inputN  Input normalDistribution object
         */
		normalDistribution(normalDistribution &&inputN);

		/*!
         * \brief Move assignment operator
         * 
         * \param inputN  Input normalDistribution object
         * \return normalDistribution& 
         */
		normalDistribution &operator=(normalDistribution &&inputN);

		/*!
         * \brief Random numbers x according to Normal (or Gaussian) random number distribution
         * The result type generated by the generator is undefined if @T is not one of float, 
         * double, or long double
         * 
         * \return Random numbers x according to Normal (or Gaussian) random number distribution
         * 
         */
		inline T operator()();

		/*!
         * \brief Random numbers x according to Normal (or Gaussian) random number distribution
         * The result type generated by the generator is undefined if @T is not one of float, 
         * double, or long double
         * 
         * \return Random numbers x according to Normal (or Gaussian) random number distribution
         * 
         */
		inline T dist();

	  private:
		// Make it noncopyable
		normalDistribution(normalDistribution const &) = delete;

		// Make it not assignable
		normalDistribution &operator=(normalDistribution const &) = delete;

	  private:
		//! Random numbers according to the Normal (or Gaussian) random number distribution
		std::normal_distribution<T> d;
	};

	/*! \class NormalDistribution
     * \brief Generates random numbers according to the Normal (or Gaussian) random number distribution
     * 
     * NOTE: This can be called without setting the State of psrandom object
     *
     */
	class NormalDistribution
	{
	  public:
		/*!
         * \brief Construct a new NormalDistribution object (default mean = 0, stddev = 1)
         * 
         * \param mean    Mean
         * \param stddev  Standard deviation
         * 
         */
		NormalDistribution(T mean = T{}, T stddev = T{1});

		/*!
         * \brief Move constructor, construct a new NormalDistribution object from input NormalDistribution object
         * 
         * \param inputN  Input NormalDistribution object
         */
		NormalDistribution(NormalDistribution &&inputN);

		/*!
         * \brief Move assignment operator
         * 
         * \param inputN  Input NormalDistribution object
         * \return NormalDistribution& 
         */
		NormalDistribution &operator=(NormalDistribution &&inputN);

		/*!
         * \brief Random numbers x according to Normal (or Gaussian) random number distribution
         * The result type generated by the generator is undefined if @T is not one of float, 
         * double, or long double
         * 
         * \return Random numbers x according to Normal (or Gaussian) random number distribution
         * 
         */
		inline T operator()();

		/*!
         * \brief Random numbers x according to Normal (or Gaussian) random number distribution
         * The result type generated by the generator is undefined if @T is not one of float, 
         * double, or long double
         * 
         * \return Random numbers x according to Normal (or Gaussian) random number distribution
         * 
         */
		inline T dist();

	  private:
		// Make it noncopyable
		NormalDistribution(NormalDistribution const &) = delete;

		// Make it not assignable
		NormalDistribution &operator=(NormalDistribution const &) = delete;

	  private:
		//! Random number engine based on Mersenne Twister algorithm.
		std::mt19937 gen;

		//! Random numbers according to the Normal (or Gaussian) random number distribution
		std::normal_distribution<T> d;
	};

	/*! \class lognormalDistribution
     * \brief Generates random numbers x > 0 according to the lognormal_distribution
     * 
     * NOTE: This should be called after setting the State of psrandom object
     * 
     */
	class lognormalDistribution
	{
	  public:
		/*!
         * \brief Construct a new lognormalDistribution object (default mean = 0, stddev = 1)
         * 
         * \param mean    Mean
         * \param stddev  Standard deviation
         * 
         */
		lognormalDistribution(T mean = T{}, T stddev = T{1});

		/*!
         * \brief Move constructor, construct a new lognormalDistribution object from input lognormalDistribution object
         * 
         * \param inputN  Input lognormalDistribution object
         */
		lognormalDistribution(lognormalDistribution &&inputN);

		/*!
         * \brief Move assignment operator
         * 
         * \param inputN  Input lognormalDistribution object
         * \return lognormalDistribution& 
         */
		lognormalDistribution &operator=(lognormalDistribution &&inputN);

		/*!
         * \brief Random numbers x according to Normal (or Gaussian) random number distribution
         * The result type generated by the generator is undefined if @T is not one of float, 
         * double, or long double
         * 
         * \return Random numbers x according to Normal (or Gaussian) random number distribution
         * 
         */
		inline T operator()();

		/*!
         * \brief Random numbers x according to Normal (or Gaussian) random number distribution
         * The result type generated by the generator is undefined if @T is not one of float, 
         * double, or long double
         * 
         * \return Random numbers x according to Normal (or Gaussian) random number distribution
         * 
         */
		inline T dist();

	  private:
		// Make it noncopyable
		lognormalDistribution(lognormalDistribution const &) = delete;

		// Make it not assignable
		lognormalDistribution &operator=(lognormalDistribution const &) = delete;

	  private:
		//! Lognormal_distribution random number distribution
		std::lognormal_distribution<T> d;
	};

	/*! \class logNormalDistribution
     * \brief Generates random numbers x > 0 according to the lognormal_distribution
     * 
     * NOTE: This can be called without setting the State of psrandom object
     * 
     */
	class logNormalDistribution
	{
	  public:
		/*!
         * \brief Construct a new logNormalDistribution object (default mean = 0, stddev = 1)
         * 
         * \param mean    Mean
         * \param stddev  Standard deviation
         * 
         */
		logNormalDistribution(T mean = T{}, T stddev = T{1});

		/*!
         * \brief Move constructor, construct a new logNormalDistribution object from input logNormalDistribution object
         * 
         * \param inputN  Input logNormalDistribution object
         */
		logNormalDistribution(logNormalDistribution &&inputN);

		/*!
         * \brief Move assignment operator
         * 
         * \param inputN  Input logNormalDistribution object
         * \return logNormalDistribution& 
         */
		logNormalDistribution &operator=(logNormalDistribution &&inputN);

		/*!
         * \brief Random numbers x according to Normal (or Gaussian) random number distribution
         * The result type generated by the generator is undefined if @T is not one of float, 
         * double, or long double
         * 
         * \return Random numbers x according to Normal (or Gaussian) random number distribution
         * 
         */
		inline T operator()();

		/*!
         * \brief Random numbers x according to Normal (or Gaussian) random number distribution
         * The result type generated by the generator is undefined if @T is not one of float, 
         * double, or long double
         * 
         * \return Random numbers x according to Normal (or Gaussian) random number distribution
         * 
         */
		inline T dist();

	  private:
		// Make it noncopyable
		logNormalDistribution(logNormalDistribution const &) = delete;

		// Make it not assignable
		logNormalDistribution &operator=(logNormalDistribution const &) = delete;

	  private:
		//! Random number engine based on Mersenne Twister algorithm.
		std::mt19937 gen;

		//! Lognormal_distribution random number distribution
		std::lognormal_distribution<T> d;
	};

	/*! \class multivariatenormalDistribution
     * \brief Multivariate normal distribution
     * 
     * NOTE: This should be called after setting the State of psrandom object
     * 
     */
	class multivariatenormalDistribution
	{
	  public:
		/*!
         * \brief constructor
         *
         * \param imean        Mean vector of size \f$n\f$
         * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
         */
		multivariatenormalDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance);

		/*!
         * \brief constructor
         * 
         * \param imean        Input mean vector of size \f$n\f$
         * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
         * \param n            Vector size
         */
		multivariatenormalDistribution(T const *imean, T const *icovariance, int const n);

		/*!
         * \brief constructor (default mean = 0)
         *
         * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
         */
		explicit multivariatenormalDistribution(EMatrixX<T> const &icovariance);

		/*!
         * \brief constructor (default mean = 0)
         * 
         * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
         * \param n            Vector size
         */
		multivariatenormalDistribution(T const *icovariance, int const n);

		/*!
         * \brief constructor (default mean = 0, covariance=I)
         * 
         * \param n vector size
         */
		explicit multivariatenormalDistribution(int const n);

		/*!
         * \brief Move constructor, construct a new multivariatenormalDistribution object from input multivariatenormalDistribution object
         * 
         * \param inputN  Input multivariatenormalDistribution object
         */
		multivariatenormalDistribution(multivariatenormalDistribution &&inputN);

		/*!
         * \brief Move assignment operator
         * 
         * \param inputN  Input multivariatenormalDistribution object
         * \return multivariatenormalDistribution& 
         */
		multivariatenormalDistribution &operator=(multivariatenormalDistribution &&inputN);

		/*!
         * \returns a vector with multivariate normal distribution
         */
		EVectorX<T> operator()();

		/*!
         * \returns a vector with multivariate normal distribution
         */
		EVectorX<T> dist();

	  private:
		// Make it noncopyable
		multivariatenormalDistribution(multivariatenormalDistribution const &) = delete;

		// Make it not assignable
		multivariatenormalDistribution &operator=(multivariatenormalDistribution const &) = delete;

	  public:
		//! Vector of size \f$n\f$
		EVectorX<T> mean;

		//! Variance-covariance matrix of size \f$ n \times n \f$
		EMatrixX<T> covariance;

	  private:
		//! Matrix of size \f$n \times n\f$
		EMatrixX<T> transform;

	  public:
		//! LU decomposition of a matrix with complete pivoting
		Eigen::FullPivLU<EMatrixX<T>> lu;

	  private:
		//! Generates random numbers according to the Normal (or Gaussian) random number distribution
		std::normal_distribution<T> d;
	};

	/*! \class multivariatenormalDistribution
     * \brief Multivariate normal distribution
     * 
     * NOTE: This can be called without setting the State of psrandom object
     * 
     */
	class multivariateNormalDistribution
	{
	  public:
		/*!
         * \brief constructor
         *
         * \param imean        Mean vector of size \f$n\f$
         * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
         */
		multivariateNormalDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance);

		/*!
         * \brief constructor
         * 
         * \param imean        Input mean vector of size \f$n\f$
         * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
         * \param n            Vector size
         */
		multivariateNormalDistribution(T const *imean, T const *icovariance, int const n);

		/*!
         * \brief constructor (default mean = 0)
         *
         * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
         */
		explicit multivariateNormalDistribution(EMatrixX<T> const &icovariance);

		/*!
         * \brief constructor (default mean = 0)
         * 
         * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
         * \param n            Vector size
         */
		multivariateNormalDistribution(T const *icovariance, int const n);

		/*!
         * \brief constructor (default mean = 0, covariance=I)
         * 
         * \param n vector size
         */
		explicit multivariateNormalDistribution(int const n);

		/*!
         * \brief Move constructor, construct a new multivariateNormalDistribution object from input multivariateNormalDistribution object
         * 
         * \param inputN  Input multivariateNormalDistribution object
         */
		multivariateNormalDistribution(multivariateNormalDistribution &&inputN);

		/*!
         * \brief Move assignment operator
         * 
         * \param inputN  Input multivariateNormalDistribution object
         * \return multivariateNormalDistribution& 
         */
		multivariateNormalDistribution &operator=(multivariateNormalDistribution &&inputN);

		/*!
         * \returns a vector with multivariate normal distribution
         */
		EVectorX<T> operator()();

		/*!
         * \returns a vector with multivariate normal distribution
         */
		EVectorX<T> dist();

	  private:
		// Make it noncopyable
		multivariateNormalDistribution(multivariateNormalDistribution const &) = delete;

		// Make it not assignable
		multivariateNormalDistribution &operator=(multivariateNormalDistribution const &) = delete;

	  public:
		//! Vector of size \f$n\f$
		EVectorX<T> mean;

		//! Variance-covariance matrix of size \f$n \times n\f$
		EMatrixX<T> covariance;

	  private:
		//! Matrix of size \f$n \times n\f$
		EMatrixX<T> transform;

	  public:
		//! LU decomposition of a matrix with complete pivoting
		Eigen::FullPivLU<EMatrixX<T>> lu;

	  private:
		//! A random number engine based on Mersenne Twister algorithm
		std::mt19937 gen;

		//! Generates random numbers according to the Normal (or Gaussian) random number distribution
		std::normal_distribution<T> d;
	};

  public:
	//! Normal (or Gaussian) random number distribution (NOTE: This should be used after setting the State of psrandom object)
	std::unique_ptr<normalDistribution> normal;

	//! Normal (or Gaussian) random number distribution (NOTE: This can be used without setting the State of psrandom object)
	std::unique_ptr<NormalDistribution> Normal;

	//! lognormal_distribution random number distribution (NOTE: This should be used after setting the State of psrandom object)
	std::unique_ptr<lognormalDistribution> lnormal;

	//! lognormal_distribution random number distribution (NOTE: This can be used without setting the State of psrandom object)
	std::unique_ptr<logNormalDistribution> lNormal;

	//! lognormal_distribution random number distribution (NOTE: This should be used after setting the State of psrandom object)
	std::unique_ptr<multivariatenormalDistribution> mvnormal;

	//! lognormal_distribution random number distribution (NOTE: This can be used without setting the State of psrandom object)
	std::unique_ptr<multivariateNormalDistribution> mvNormal;

  public:
	/*!
     * \brief Replaces the normal object 
     * 
     * \param imean    Input Mean
     * \param istddev  Input standard deviation
     * 
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_normal(T imean, T istddev);

	/*!
     * \brief Replaces the Normal object 
     * 
     * \param imean    Input Mean
     * \param istddev  Input standard deviation
     * 
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_Normal(T imean, T istddev);

	/*!
     * \brief Replaces the lnormal object 
     * 
     * \param imean    Input Mean
     * \param istddev  Input standard deviation
     * 
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_lnormal(T imean, T istddev);

	/*!
     * \brief Replaces the lNormal object 
     * 
     * \param imean    Input Mean
     * \param istddev  Input standard deviation
     * 
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_lNormal(T imean, T istddev);

	/*!
     * \brief Replaces the mvnormal object 
     * 
     * \param imean        Mean vector of size \f$n\f$
     * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
     * 
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_mvnormal(EVectorX<T> const &imean, EMatrixX<T> const &icovariance);

	/*!
     * \brief Replaces the mvnormal object
     *
     * \param imean        Input mean vector of size \f$n\f$
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     *
     * \return true 
     * \return false in failure to allocate storage
     */

	inline bool set_mvnormal(T const *imean, T const *icovariance, int const n);

	/*!
     * \brief Replaces the mvnormal object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     *
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_mvnormal(EMatrixX<T> const &icovariance);

	/*!
     * \brief Replaces the mvnormal object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     *
     * \return true 
     * \return false in failure to allocate storage
     */

	inline bool set_mvnormal(T const *icovariance, int const n);

	/*!
     * \brief Replaces the mvnormal object (default mean = 0, covariance=I)
     *
     * \param n vector size
     *
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_mvnormal(int const n);

	/*!
     * \brief Replaces the mvNormal object 
     * 
     * \param imean        Mean vector of size \f$n\f$
     * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
     * 
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_mvNormal(EVectorX<T> const &imean, EMatrixX<T> const &icovariance);

	/*!
     * \brief Replaces the mvNormal object
     *
     * \param imean        Input mean vector of size \f$n\f$
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     *
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_mvNormal(T const *imean, T const *icovariance, int const n);

	/*!
     * \brief Replaces the mvNormal object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     *
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_mvNormal(EMatrixX<T> const &icovariance);

	/*!
     * \brief Replaces the mvNormal object (default mean = 0)
     *
     * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
     * \param n            Vector size
     *
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_mvNormal(T const *icovariance, int const n);

	/*!
     * \brief Replaces the mvNormal object (default mean = 0, covariance=I)
     *
     * \param n vector size
     *
     * \return true 
     * \return false in failure to allocate storage
     */
	inline bool set_mvNormal(int const n);
};

/*!
 * \brief Construct a new psrandom::psrandom object
 * 
 * It sets the seed to random number
 * 
 */
template <typename T>
psrandom<T>::psrandom()
{
	std::lock_guard<std::mutex> lock(PRNG_m);
	if (PRNG_seed == 0)
	{
		PRNG_seed = std::random_device{}();
	}
};

/*!
 * \brief Construct a new psrandom object
 * It sets the seed to iseed
 * 
 * \param iseed Input seed for random number seed
 * 
 */
template <typename T>
psrandom<T>::psrandom(int const iseed)
{
	std::lock_guard<std::mutex> lock(PRNG_m);
	if (PRNG_seed == 0)
	{
		PRNG_seed = static_cast<std::size_t>(iseed);
	}
}

/*!
 * \brief init Task on each node to set the current state of the engine
 * 
 * NOTE: You should not call this function directly
 * 
 */
template <typename T>
void psrandom<T>::init_Task()
{
	std::vector<std::size_t> rseed(std::mt19937::state_size);

	// Get the local number of workers
	std::size_t nlocalworkers = static_cast<std::size_t>(torc_i_num_workers());

	// Node Id (MPI rank)
	std::size_t node_id = static_cast<std::size_t>(torc_node_id());

	std::size_t n = nlocalworkers * (node_id + 1);

	for (std::size_t i = 0; i < nlocalworkers; i++)
	{
		std::size_t const j = PRNG_seed + n + i;
		std::iota(rseed.begin(), rseed.end(), j);

		// Seed the engine with unsigned ints
		std::seed_seq sseq(rseed.begin(), rseed.end());

		// For each thread feed the RNG
		NumberGenerator[i].seed(sseq);

		Saru s(PRNG_seed, n, i);
		saru[i] = std::move(s);
	}
}

/*!
 * \brief Set the State of psrandom object
 * 
 * \return \a true when sets the current state of the engine successfully 
 */
template <typename T>
bool psrandom<T>::init()
{
	{
		// Make sure MPI is initilized
		auto initialized = 0;
		MPI_Initialized(&initialized);
		if (!initialized)
		{
			UMUQFAILRETURN("Failed to initilize MPI!");
		}
	}

	{
		std::lock_guard<std::mutex> lock(PRNG_m);

		// Check if psrandom is already initilized
		if (PRNG_initialized)
		{
			return true;
		}

		PRNG_initialized = true;
	}

	torc_register_task((void *)psrandom<T>::init_Task);

	int const nlocalworkers = torc_i_num_workers();

	try
	{
		NumberGenerator.resize(nlocalworkers);
		saru.resize(nlocalworkers);
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	};

	for (int i = 0; i < torc_num_nodes(); i++)
	{
		torc_create_ex(i * nlocalworkers, 1, (void (*)())psrandom<T>::init_Task, 0);
	}
	torc_waitall();

	return true;
}

/*!
 * \brief Set the State of psrandom object
 * 
 * \return \a true when sets the current state of the engine successfully 
 */
template <typename T>
bool psrandom<T>::setState()
{
	return psrandom<T>::init();
}

/*!
 * \brief Set the Seed object
 * NOTE: Seeding of the PRNG can be done only once globally either in 
 * construction or through setting the seed
 * 
 * \param iseed  Input seed for random number initialization 
 * 
 * \return true  when sets the seed of the engine successfully
 * \return false If the PRNG is already initialized or the state has been set
 */
inline bool setSeed(int const iseed)
{
	std::lock_guard<std::mutex> lock(PRNG_m);
	if (!PRNG_initialized)
	{
		PRNG_seed = static_cast<std::size_t>(iseed);
		return true;
	}
	// It has been initialized before
	return false;
}

/*!
 * \brief Uniform random number between \f$ [a \cdots b) \f$
 * Data type one of float, double
 * 
 * \param a lower bound 
 * \param b upper bound
 * 
 * NOTE: This should be called after setting the State of psrandom object
 * 
 * Advance the PRNG state by 1, and output a T precision \f$ [a \cdots b) \f$ number (default \f$ a = 0 , b = 1 \f$)
 * This is a partial specialization to make a special case for double precision uniform random number
 */
template <typename T>
inline T psrandom<T>::unirnd(T const a, T const b)
{
	UMUQFAIL("The Uniform random number of requested type is not implemented!");
}

/*!
 * \brief Uniform random number between \f$ [a \cdots b) \f$
 *
 * \param a lower bound 
 * \param b upper bound
 * 
 * NOTE: This should be called after setting the State of psrandom object
 *
 * Advance the PRNG state by 1, and output a T precision \f$ [a \cdots b) \f$ number (default \f$ a = 0 , b = 1 \f$)
 * This is a partial specialization to make a special case for double precision uniform random number
 */
template <>
inline double psrandom<double>::unirnd(double const a, double const b)
{
	// Get the thread ID
	int const me = torc_i_worker_id();
	return saru[me].d(a, b);
}

/*!
 * \brief This is a partial specialization to make a special case for float precision uniform random number between \f$ [a \cdots b) \f$
 * 
 * \param a lower bound 
 * \param b upper bound
 *
 * NOTE: This should be called after setting the State of psrandom object
 * 
 */
template <>
inline float psrandom<float>::unirnd(float a, float b)
{
	// Get the thread ID
	int const me = torc_i_worker_id();
	return saru[me].f(a, b);
}

/*!
 * \brief Advance state by 1, and output a double precision \f$ [0 \cdots 1) \f$ floating point
 * 
 * \returns a uniform random number of a double precision \f$ [0 \cdots 1) \f$ floating point 
 * 
 * Reference:
 * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, Comput. Phys. Comm. 184 (2013), 1119–1128.
 */
template <typename T>
inline double psrandom<T>::d() { return saru[0].d(); }

/*!
 * \returns a uniform random number of a single precision \f$ [0 \cdots 1) \f$ floating point
 * 
 * \Advance state by 1, and output a single precision \f$ [0 \cdots 1) \f$ floating point
 * 
 * Reference:
 * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, Comput. Phys. Comm. 184 (2013), 1119–1128.
 */
template <typename T>
inline float psrandom<T>::f() { return saru[0].f(); }

/*!
 * \returns an unisgned 32 bit integer pseudo-random value
 * 
 * \brief Advance state by 1, and output a 32 bit integer pseudo-random value.
 * 
 * Reference:
 * Y. Afshar, F. Schmid, A. Pishevar, S. Worley, Comput. Phys. Comm. 184 (2013), 1119–1128.
 */
template <typename T>
inline unsigned int psrandom<T>::u32() { return saru[0].u32(); }

/*!
 * \brief The Fisher-Yates shuffle is used to permute randomly given input array.
 *
 * \tparam D Input data type
 * 
 * \param idata  Array of input data
 * \param nSize  Size of the array idata
 *  
 * NOTE: This should be called after setting the State of psrandom object
 *
 * The permutations generated by this algorithm occur with the same probability.
 *
 * References : 
 * R. Durstenfeld, "Algorithm 235: Random permutation" Communications of the ACM, 7 (1964), p. 420
 * 
 */
template <typename T>
template <typename D>
inline void psrandom<T>::shuffle(D *idata, int const nSize)
{
	// Get the thread ID
	int const me = torc_i_worker_id();

	for (int i = nSize - 1; i > 0; --i)
	{
		unsigned int const idx = saru[me].u32(i);
		std::swap(idata[i], idata[idx]);
	}
}

/*!
 * \brief The Fisher-Yates shuffle is used to permute randomly given input array.
 * 
 * \tparam D Input data type
 * 
 * \param idata  Array of input data
 * \param nSize  Size of the array idata
 * 
 * NOTE: This can be called without setting the State of psrandom object
 * 
 */
template <typename T>
template <typename D>
inline void psrandom<T>::Shuffle(D *idata, int const nSize)
{
	for (int i = nSize - 1; i > 0; --i)
	{
		unsigned int const idx = saru[0].u32(i);
		std::swap(idata[i], idata[idx]);
	}
}

/*! \fn multinomial
 * \brief The multinomial distribution
 * 
 * \tparam   T        Data type 
 * 
 * \param    p        Vector of probabilities \f$ p_1, \cdots, p_k \f$
 * \param    K        Size of vector which shows K possible mutually exclusive outcomes 
 * \param    N        N independent trials
 * \param    mndist   A random sample from the multinomial distribution (with size of K)
 *  
 * NOTE: This should be called after setting the State of psrandom object
 * 
 * 
 * Let \f$ X=\left( X_1, \cdots, X_K \right) \f$ have a multinomial distribution \f$ M_K\left(N, p\right) \f$
 * The distribution of \f$ X \f$ is given by:
 * \f[
 *     Pr(X_1=n_1, \cdots, X_K=n_K) = \frac{N!}{\left(n_1! n_2! \cdots n_K! \right)}  p_1^{n_1}  p_2^{n_2} \cdots p_K^{n_K}
 * \f] 
 *
 * where \f$ n_1, \cdots n_K \f$ are nonnegative integers satisfying \f$ sum_{i=1}^{K} {n_i} = N\f$,
 * and \f$p = \left(p_1, \cdots, p_K\right)\f$ is a probability distribution. 
 *
 * Random variates are generated using the conditional binomial method.
 * This scales well with N and does not require a setup step.
 *   
 *  Ref: 
 *  C.S. David, The computer generation of multinomial random variates,
 *  Comp. Stat. Data Anal. 16 (1993) 205-217
 */
template <typename T>
bool psrandom<T>::multinomial(T const *p, unsigned int const K, unsigned int const N, unsigned int *mndist)
{
	if (!PRNG_initialized)
	{
		UMUQFAILRETURN("One should set the current state of the engine before calling this function!");
	}

	// Get the thread ID
	int const me = torc_i_worker_id();

	T const totpsum = std::accumulate(p, p + K, 0);

	T psum(0);
	unsigned int nsum(0);
	for (int i = 0; i < K; i++)
	{
		if (p[i] > 0.0)
		{
			std::binomial_distribution<> d(N - nsum, p[i] / (totpsum - psum));
			mndist[i] = d(NumberGenerator[me]);
		}
		else
		{
			mndist[i] = 0;
		}
		psum += p[i];
		nsum += mndist[i];
	}
	return true;
}

/*! \fn Multinomial
 * \brief The multinomial distribution
 * 
 * NOTE: This can be called without setting the State of psrandom object
 */
template <typename T>
bool psrandom<T>::Multinomial(T const *p, unsigned int const K, unsigned int const N, unsigned int *mndist)
{
	std::mt19937 gen(std::random_device{}());

	T const totpsum = std::accumulate(p, p + K, 0);

	T psum(0);
	unsigned int nsum(0);
	for (int i = 0; i < K; i++)
	{
		if (p[i] > 0.0)
		{
			std::binomial_distribution<> d(N - nsum, p[i] / (totpsum - psum));
			mndist[i] = d(gen);
		}
		else
		{
			mndist[i] = 0;
		}
		psum += p[i];
		nsum += mndist[i];
	}
	return true;
}

/*!
 * \brief Construct a new normalDistribution object (default mean = 0, stddev = 1)
 * 
 * \param mean    Mean
 * \param stddev  Standard deviation
 * 
 */
template <typename T>
psrandom<T>::normalDistribution::normalDistribution(T mean, T stddev) : d(mean, stddev)
{
	if (!PRNG_initialized)
	{
		UMUQFAIL("One should set the current state of the engine before constructing this object!");
	}
}

/*!
 * \brief Move constructor, construct a new normalDistribution object from input normalDistribution object
 * 
 * \param inputN  Input normalDistribution object
 */
template <typename T>
psrandom<T>::normalDistribution::normalDistribution(psrandom<T>::normalDistribution &&inputN)
{
	this->d = std::move(inputN.d);
}

/*!
 * \brief Move assignment operator
 * 
 * \param inputN  Input normalDistribution object
 * \return normalDistribution& 
 */
template <typename T>
typename psrandom<T>::normalDistribution &psrandom<T>::normalDistribution::operator=(psrandom<T>::normalDistribution &&inputN)
{
	this->d = std::move(inputN.d);
	return *this;
}

/*!
 * \brief Random numbers x according to Normal (or Gaussian) random number distribution
 * The result type generated by the generator is undefined if @T is not one of float, 
 * double, or long double
 * 
 * \return Random numbers x according to Normal (or Gaussian) random number distribution
 * 
 */
template <typename T>
inline T psrandom<T>::normalDistribution::operator()()
{
	// Get the thread ID
	int const me = torc_i_worker_id();
	return this->d(NumberGenerator[me]);
}

/*!
 * \brief Random numbers x according to Normal (or Gaussian) random number distribution
 * The result type generated by the generator is undefined if @T is not one of float, 
 * double, or long double
 * 
 * \return Random numbers x according to Normal (or Gaussian) random number distribution
 * 
 */
template <typename T>
inline T psrandom<T>::normalDistribution::dist()
{
	// Get the thread ID
	int const me = torc_i_worker_id();
	return this->d(NumberGenerator[me]);
}

/*!
 * \brief Construct a new NormalDistribution<T>::NormalDistribution object (default mean = 0, stddev = 1)
 * 
 * \param mean    Mean
 * \param stddev  Standard deviation
 * 
 */
template <typename T>
psrandom<T>::NormalDistribution::NormalDistribution(T mean, T stddev) : gen(std::random_device{}()), d(mean, stddev) {}

/*!
 * \brief Move constructor, construct a new NormalDistribution object from input NormalDistribution object
 * 
 * \param inputN  Input NormalDistribution object
 */
template <typename T>
psrandom<T>::NormalDistribution::NormalDistribution(psrandom<T>::NormalDistribution &&inputN)
{
	this->gen = std::move(inputN.gen);
	this->d = std::move(inputN.d);
}

/*!
 * \brief Move assignment operator
 * 
 * \param inputN  Input NormalDistribution object
 * \return NormalDistribution& 
 */
template <typename T>
typename psrandom<T>::NormalDistribution &psrandom<T>::NormalDistribution::operator=(psrandom<T>::NormalDistribution &&inputN)
{
	this->gen = std::move(inputN.gen);
	this->d = std::move(inputN.d);
	return *this;
}

/*!
 * \brief Random numbers x according to Normal (or Gaussian) random number distribution
 * The result type generated by the generator is undefined if @T is not one of float, 
 * double, or long double
 * 
 * \return Random numbers x according to Normal (or Gaussian) random number distribution
 * 
 */
template <typename T>
inline T psrandom<T>::NormalDistribution::operator()() { return this->d(this->gen); }

/*!
 * \brief Random numbers x according to Normal (or Gaussian) random number distribution
 * The result type generated by the generator is undefined if @T is not one of float, 
 * double, or long double
 * 
 * \return Random numbers x according to Normal (or Gaussian) random number distribution
 * 
 */
template <typename T>
inline T psrandom<T>::NormalDistribution::dist() { return this->d(this->gen); }

/*!
 * \brief Construct a new lognormalDistribution object (default mean = 0, stddev = 1)
 * 
 * \param mean    Mean
 * \param stddev  Standard deviation
 * 
 */
template <typename T>
psrandom<T>::lognormalDistribution::lognormalDistribution(T mean, T stddev) : d(mean, stddev)
{
	if (!PRNG_initialized)
	{
		UMUQFAIL("One should set the current state of the engine before constructing this object!");
	}
}

/*!
 * \brief Move constructor, construct a new lognormalDistribution object from input lognormalDistribution object
 * 
 * \param inputN  Input lognormalDistribution object
 */
template <typename T>
psrandom<T>::lognormalDistribution::lognormalDistribution(psrandom<T>::lognormalDistribution &&inputN)
{
	this->d = std::move(inputN.d);
}

/*!
 * \brief Move assignment operator
 * 
 * \param inputN  Input lognormalDistribution object
 * \return lognormalDistribution& 
 */
template <typename T>
typename psrandom<T>::lognormalDistribution &psrandom<T>::lognormalDistribution::operator=(psrandom<T>::lognormalDistribution &&inputN)
{
	this->d = std::move(inputN.d);
	return *this;
}

/*!
 * \brief Random numbers x according to Normal (or Gaussian) random number distribution
 * The result type generated by the generator is undefined if @T is not one of float, 
 * double, or long double
 * 
 * \return Random numbers x according to Normal (or Gaussian) random number distribution
 * 
 */
template <typename T>
inline T psrandom<T>::lognormalDistribution::operator()()
{
	// Get the thread ID
	int const me = torc_i_worker_id();
	return this->d(NumberGenerator[me]);
}

/*!
 * \brief Random numbers x according to Normal (or Gaussian) random number distribution
 * The result type generated by the generator is undefined if @T is not one of float, 
 * double, or long double
 * 
 * \return Random numbers x according to Normal (or Gaussian) random number distribution
 * 
 */
template <typename T>
inline T psrandom<T>::lognormalDistribution::dist()
{
	// Get the thread ID
	int const me = torc_i_worker_id();
	return this->d(NumberGenerator[me]);
}

/*!
 * \brief Construct a new logNormalDistribution object (default mean = 0, stddev = 1)
 * 
 * \param mean    Mean
 * \param stddev  Standard deviation
 * 
 */
template <typename T>
psrandom<T>::logNormalDistribution::logNormalDistribution(T mean, T stddev) : gen(std::random_device{}()), d(mean, stddev) {}

/*!
 * \brief Move constructor, construct a new logNormalDistribution object from input logNormalDistribution object
 * 
 * \param inputN  Input logNormalDistribution object
 */
template <typename T>
psrandom<T>::logNormalDistribution::logNormalDistribution(psrandom<T>::logNormalDistribution &&inputN)
{
	this->gen = std::move(inputN.gen);
	this->d = std::move(inputN.d);
}

/*!
 * \brief Move assignment operator
 * 
 * \param inputN  Input logNormalDistribution object
 * \return logNormalDistribution& 
 */
template <typename T>
typename psrandom<T>::logNormalDistribution &psrandom<T>::logNormalDistribution::operator=(psrandom<T>::logNormalDistribution &&inputN)
{
	this->gen = std::move(inputN.gen);
	this->d = std::move(inputN.d);
	return *this;
}

/*!
 * \brief Random numbers x according to the lognormal_distribution distribution
 * The result type generated by the generator is undefined if @T is not one of float, 
 * double, or long double
 * 
 * \return Random random number \f$ x > 0 \f$ according to the lognormal_distribution
 * 
 */
template <typename T>
inline T psrandom<T>::logNormalDistribution::operator()()
{
	return this->d(this->gen);
}

/*!
 * \brief Random numbers x according to the lognormal_distribution distribution
 * The result type generated by the generator is undefined if @T is not one of float, 
 * double, or long double
 * 
 * \return Random random number \f$ x > 0 \f$ according to the lognormal_distribution
 * 
 */
template <typename T>
inline T psrandom<T>::logNormalDistribution::dist()
{
	return this->d(this->gen);
}

/*!
 * \brief constructor
 *
 * \param imean        Mean vector of size \f$n\f$
 * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T>
psrandom<T>::multivariatenormalDistribution::multivariatenormalDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance) : mean(imean),
																																		covariance(icovariance),
																																		lu(icovariance)
{
	if (!PRNG_initialized)
	{
		UMUQFAIL("One should set the current state of the engine before constructing this object!");
	}

	// Computes eigenvalues and eigenvectors of selfadjoint matrices.
	Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(this->covariance);
	this->transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor
 * 
 * \param imean        Input mean vector of size \f$n\f$
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
psrandom<T>::multivariatenormalDistribution::multivariatenormalDistribution(T const *imean, T const *icovariance, int const n) : mean(CTEMapX<T>(imean, n, 1)),
																																 covariance(CTEMapX<T>(icovariance, n, n)),
																																 lu(CTEMapX<T>(icovariance, n, n))
{
	if (!PRNG_initialized)
	{
		UMUQFAIL("One should set the current state of the engine before constructing this object!");
	}

	// Computes eigenvalues and eigenvectors of selfadjoint matrices.
	Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(this->covariance);
	this->transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T>
psrandom<T>::multivariatenormalDistribution::multivariatenormalDistribution(EMatrixX<T> const &icovariance) : multivariatenormalDistribution(EVectorX<T>::Zero(icovariance.rows()), icovariance)
{
	if (!PRNG_initialized)
	{
		UMUQFAIL("One should set the current state of the engine before constructing this object!");
	}
}

/*!
 * \brief constructor (default mean = 0)
 * 
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
psrandom<T>::multivariatenormalDistribution::multivariatenormalDistribution(T const *icovariance, int const n) : mean(EVectorX<T>::Zero(n)),
																												 covariance(CTEMapX<T>(icovariance, n, n)),
																												 lu(CTEMapX<T>(icovariance, n, n))
{
	if (!PRNG_initialized)
	{
		UMUQFAIL("One should set the current state of the engine before constructing this object!");
	}

	// Computes eigenvalues and eigenvectors of selfadjoint matrices.
	Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(this->covariance);
	this->transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor (default mean = 0, covariance=I)
 * 
 * \param n vector size
 */
template <typename T>
psrandom<T>::multivariatenormalDistribution::multivariatenormalDistribution(int const n) : mean(EVectorX<T>::Zero(n)),
																						   covariance(EMatrixX<T>::Identity(n, n)),
																						   transform(EMatrixX<T>::Identity(n, n)),
																						   lu(EMatrixX<T>::Identity(n, n))
{
	if (!PRNG_initialized)
	{
		UMUQFAIL("One should set the current state of the engine before constructing this object!");
	}
}

/*!
 * \brief Move constructor, construct a new multivariatenormalDistribution object from input multivariatenormalDistribution object
 * 
 * \param inputN  Input multivariatenormalDistribution object
 */
template <typename T>
psrandom<T>::multivariatenormalDistribution::multivariatenormalDistribution(psrandom<T>::multivariatenormalDistribution &&inputN)
{
	this->mean = std::move(inputN.mean);
	this->covariance = std::move(inputN.covariance);
	this->transform = std::move(inputN.transform);
	this->lu = std::move(inputN.lu);
	this->d = std::move(inputN.d);
}

/*!
 * \brief Move assignment operator
 * 
 * \param inputN  Input multivariatenormalDistribution object
 * \return multivariatenormalDistribution& 
 */
template <typename T>
typename psrandom<T>::multivariatenormalDistribution &psrandom<T>::multivariatenormalDistribution::operator=(psrandom<T>::multivariatenormalDistribution &&inputN)
{
	this->mean = std::move(inputN.mean);
	this->covariance = std::move(inputN.covariance);
	this->transform = std::move(inputN.transform);
	this->lu = std::move(inputN.lu);
	this->d = std::move(inputN.d);
	return *this;
}

/*!
 * \returns a vector with multivariate normal distribution
 */
template <typename T>
EVectorX<T> psrandom<T>::multivariatenormalDistribution::operator()()
{
	int const me = torc_i_worker_id();
	return this->mean + this->transform * EVectorX<T>{this->mean.size()}.unaryExpr([&](T const x) { return this->d(NumberGenerator[me]); });
}

/*!
 * \returns a vector with multivariate normal distribution
 */
template <typename T>
EVectorX<T> psrandom<T>::multivariatenormalDistribution::dist()
{
	int const me = torc_i_worker_id();
	return this->mean + this->transform * EVectorX<T>{this->mean.size()}.unaryExpr([&](T const x) { return this->d(NumberGenerator[me]); });
}

/*!
 * \brief constructor
 *
 * \param imean        Mean vector of size \f$n\f$
 * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T>
psrandom<T>::multivariateNormalDistribution::multivariateNormalDistribution(EVectorX<T> const &imean, EMatrixX<T> const &icovariance) : mean(imean),
																																		covariance(icovariance),
																																		lu(icovariance),
																																		gen(std::random_device{}())
{
	// Computes eigenvalues and eigenvectors of selfadjoint matrices.
	Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(this->covariance);
	this->transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor
 * 
 * \param imean        Input mean vector of size \f$n\f$
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
psrandom<T>::multivariateNormalDistribution::multivariateNormalDistribution(T const *imean, T const *icovariance, int const n) : mean(CTEMapX<T>(imean, n, 1)),
																																 covariance(CTEMapX<T>(icovariance, n, n)),
																																 lu(CTEMapX<T>(icovariance, n, n)),
																																 gen(std::random_device{}())
{
	// Computes eigenvalues and eigenvectors of selfadjoint matrices.
	Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(this->covariance);
	this->transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 */
template <typename T>
psrandom<T>::multivariateNormalDistribution::multivariateNormalDistribution(EMatrixX<T> const &icovariance) : multivariateNormalDistribution(EVectorX<T>::Zero(icovariance.rows()), icovariance) {}

/*!
 * \brief constructor (default mean = 0)
 * 
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 */
template <typename T>
psrandom<T>::multivariateNormalDistribution::multivariateNormalDistribution(T const *icovariance, int const n) : mean(EVectorX<T>::Zero(n)),
																												 covariance(CTEMapX<T>(icovariance, n, n)),
																												 lu(CTEMapX<T>(icovariance, n, n)),
																												 gen(std::random_device{}())
{
	// Computes eigenvalues and eigenvectors of selfadjoint matrices.
	Eigen::SelfAdjointEigenSolver<EMatrixX<T>> es(this->covariance);
	this->transform = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
}

/*!
 * \brief constructor (default mean = 0, covariance=I)
 * 
 * \param n vector size
 */
template <typename T>
psrandom<T>::multivariateNormalDistribution::multivariateNormalDistribution(int const n) : mean(EVectorX<T>::Zero(n)),
																						   covariance(EMatrixX<T>::Identity(n, n)),
																						   transform(EMatrixX<T>::Identity(n, n)),
																						   lu(EMatrixX<T>::Identity(n, n)),
																						   gen(std::random_device{}()) {}

/*!
 * \brief Move constructor, construct a new multivariateNormalDistribution object from input multivariateNormalDistribution object
 * 
 * \param inputN  Input multivariateNormalDistribution object
 */
template <typename T>
psrandom<T>::multivariateNormalDistribution::multivariateNormalDistribution(psrandom<T>::multivariateNormalDistribution &&inputN)
{
	this->mean = std::move(inputN.mean);
	this->covariance = std::move(inputN.covariance);
	this->transform = std::move(inputN.transform);
	this->lu = std::move(inputN.lu);
	this->gen = std::move(inputN.gen);
	this->d = std::move(inputN.d);
}

/*!
 * \brief Move assignment operator
 * 
 * \param inputN  Input multivariateNormalDistribution object
 * \return multivariateNormalDistribution& 
 */
template <typename T>
typename psrandom<T>::multivariateNormalDistribution &psrandom<T>::multivariateNormalDistribution::operator=(psrandom<T>::multivariateNormalDistribution &&inputN)
{
	this->mean = std::move(inputN.mean);
	this->covariance = std::move(inputN.covariance);
	this->transform = std::move(inputN.transform);
	this->lu = std::move(inputN.lu);
	this->gen = std::move(inputN.gen);
	this->d = std::move(inputN.d);
	return *this;
}

/*!
 * \returns a vector with multivariate normal distribution
 */
template <typename T>
EVectorX<T> psrandom<T>::multivariateNormalDistribution::operator()()
{
	return this->mean + this->transform * EVectorX<T>{this->mean.size()}.unaryExpr([&](T const x) { return this->d(this->gen); });
}

/*!
 * \returns a vector with multivariate normal distribution
 */
template <typename T>
EVectorX<T> psrandom<T>::multivariateNormalDistribution::dist()
{
	return this->mean + this->transform * EVectorX<T>{this->mean.size()}.unaryExpr([&](T const x) { return this->d(this->gen); });
}

/*!
 * \brief Replaces the normal object
 *
 * \param imean    Input mean
 * \param istddev  Input standard deviation
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_normal(T imean, T istddev)
{
	try
	{
		psrandom<T>::normal.reset(new psrandom<T>::normalDistribution(imean, istddev));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the Normal object
 *
 * \param imean    Input mean
 * \param istddev  Input standard deviation
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_Normal(T imean, T istddev)
{
	try
	{
		psrandom<T>::Normal.reset(new psrandom<T>::NormalDistribution(imean, istddev));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the lnormal object
 *
 * \param imean    Input mean
 * \param istddev  Input standard deviation
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_lnormal(T imean, T istddev)
{
	try
	{
		psrandom<T>::lnormal.reset(new psrandom<T>::lognormalDistribution(imean, istddev));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the lNormal object
 *
 * \param imean    Input mean
 * \param istddev  Input standard deviation
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_lNormal(T imean, T istddev)
{
	try
	{
		psrandom<T>::lNormal.reset(new psrandom<T>::logNormalDistribution(imean, istddev));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the mvnormal object
 * 
 * \param imean        Mean vector of size \f$n\f$
 * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
 * 
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_mvnormal(EVectorX<T> const &imean, EMatrixX<T> const &icovariance)
{
	try
	{
		psrandom<T>::mvnormal.reset(new psrandom<T>::multivariatenormalDistribution(imean, icovariance));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the mvnormal object
 *
 * \param imean        Input mean vector of size \f$n\f$
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_mvnormal(T const *imean, T const *icovariance, int const n)
{
	try
	{
		psrandom<T>::mvnormal.reset(new psrandom<T>::multivariatenormalDistribution(imean, icovariance, n));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the mvnormal object (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_mvnormal(EMatrixX<T> const &icovariance)
{
	try
	{
		psrandom<T>::mvnormal.reset(new psrandom<T>::multivariatenormalDistribution(icovariance));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the mvnormal object (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_mvnormal(T const *icovariance, int const n)
{
	try
	{
		psrandom<T>::mvnormal.reset(new psrandom<T>::multivariatenormalDistribution(icovariance, n));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the mvnormal object (default mean = 0, covariance=I)
 *
 * \param n vector size
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_mvnormal(int const n)
{
	try
	{
		psrandom<T>::mvnormal.reset(new psrandom<T>::multivariatenormalDistribution(n));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the mvNormal object
 *
 * \param imean        Mean vector of size \f$n\f$
 * \param icovariance  Input Variance-covariance matrix of size \f$n \times n\f$
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_mvNormal(EVectorX<T> const &imean, EMatrixX<T> const &icovariance)
{
	try
	{
		psrandom<T>::mvNormal.reset(new psrandom<T>::multivariateNormalDistribution(imean, icovariance));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the mvNormal object
 *
 * \param imean        Input mean vector of size \f$n\f$
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_mvNormal(T const *imean, T const *icovariance, int const n)
{
	try
	{
		psrandom<T>::mvNormal.reset(new psrandom<T>::multivariateNormalDistribution(imean, icovariance, n));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the mvNormal object (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_mvNormal(EMatrixX<T> const &icovariance)
{
	try
	{
		psrandom<T>::mvNormal.reset(new psrandom<T>::multivariateNormalDistribution(icovariance));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the mvNormal object (default mean = 0)
 *
 * \param icovariance  Input variance-covariance matrix of size \f$n \times n\f$
 * \param n            Vector size
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_mvNormal(T const *icovariance, int const n)
{
	try
	{
		psrandom<T>::mvNormal.reset(new psrandom<T>::multivariateNormalDistribution(icovariance, n));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

/*!
 * \brief Replaces the mvNormal object (default mean = 0, covariance=I)
 *
 * \param n vector size
 *
 * \return true 
 * \return false in failure to allocate storage
 */
template <typename T>
inline bool psrandom<T>::set_mvNormal(int const n)
{
	try
	{
		psrandom<T>::mvNormal.reset(new psrandom<T>::multivariateNormalDistribution(n));
	}
	catch (...)
	{
		UMUQFAILRETURN("Failed to allocate memory!");
	}
	return true;
}

#endif
