#ifndef UMUQ_FITFUNCTION_H
#define UMUQ_FITFUNCTION_H

template <typename T, class F>
class fitfunction
{
  public:
    bool init()
    {
        return static_cast<F *>(this)->init();
    }

    T likelihood(T *iData, int ndimiData, T *oData, int ndimoData, int *info)
    {
        return static_cast<F *>(this)->likelihood(iData, ndimiData, oData, ndimoData, info);
    }

  private:
    friend F;
};

#endif
