#ifndef UMUQ_TMCMC_H
#define UMUQ_TMCMC_H

#include "data/stdata.hpp"
#include "data/datatype.hpp"
#include "data/runinfo.hpp"

template <typename T>
class tmcmc
{
public:
  tmcmc() : inputFilename("tmcmc.par"),
            currentData(nullptr),
            fullData(nullptr),
            expData(nullptr)
  {
  }

  bool init()
  {
    // Read the input problem size and variables from an input file
    if (Data.load(inputFilename))
    {
      {
        //Creating a temporary database
        database<T> d(Data.nDim, Data.maxGenerations);
        Data1<T> = std::move(d);

        //Set the pointer to the created databse object
        currentData = &Data1<T>;

        //Set the update Task function to be used for updating on multi threads or processors
        currentData->setTask(updateTask1<T>);

        //Initilize the update Task 
        if (!currentData->registerTask())
        {
          return false;
        }
      }

      {
        //Creating a temporary database
        database<T> d(Data.nDim, Data.maxGenerations);
        Data2<T> = std::move(d);

        //Set the pointer to the created databse object
        fullData = &Data2<T>;

        //Set the update Task function to be used for updating on multi threads or processors
        fullData->setTask(updateTask2<T>);

        //Initilize the update Task 
        if (!fullData->registerTask())
        {
          return false;
        }
      }

      if (runData.reset(Data.nDim, Data.maxGenerations))
      {


        return true;
      }


      return false;
    }
    return false;
  }

  inline void setInputFileName(std::string const &fileName = "tmcmc.par")
  {
    inputFilename = fileName;
  }

private:
  /*!
   * \brief Check to see whether the file fileName exists and accessible to read or write!
   *  
   * \returns true if the file exists 
   */
  inline bool isFileExist(const char *fileName)
  {
    struct stat buffer;
    return (stat(fileName, &buffer) == 0);
  }

public:
  //! Input file name
  std::string inputFilename;

  //! Current data
  database<T> *currentData;

  //! Full data pointer
  database<T> *fullData;

  //! Experimental data pointer
  database<T> *expData;

  //Running data
  runinfo<T> runData;

  //! stream data for getting the problem size and variables from the input file
  stdata<T> Data;
};

#endif
