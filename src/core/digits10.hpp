#ifndef UMHBM_DIGITS10_H
#define UMHBM_DIGITS10_H

// default implementation of digits10(), based on numeric_limits if specialized,
// 0 for integer types, and log10(epsilon()) otherwise.
template <typename T,
		  bool use_numeric_limits = std::numeric_limits<T>::is_specialized,
		  bool is_integer = std::numeric_limits<T>::is_integer>
struct default_digits10
{
	static int run() { return std::numeric_limits<T>::digits10; }
};

template <typename T>
struct default_digits10<T, false, false> // Floating point
{
	static int run() { return int(std::ceil(-std::log10(std::numeric_limits<T>::epsilon()))); }
};

template <typename T>
struct default_digits10<T, false, true> // Integer
{
	static int run() { return 0; }
};

template <typename T>
static inline int digits10()
{
	return default_digits10<T>::run();
}

#endif
