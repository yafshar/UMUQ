#ifndef UMUQ_DFTFE_H
#define UMUQ_DFTFE_H

#include "core/core.hpp"
#include "io/io.hpp"
#include "misc/parser.hpp"
#include "units/speciesname.hpp"
#include "units/units.hpp"

namespace umuq
{

/*! 
 * \defgroup 3rdparty Interfaces to the 3rd party codes e.g. DFT-FE, LAMMPS, ...
 */

/*! \class dftfe
 * \ingroup 3rdparty
 * 
 * \brief General class for an interface to the \c DFT-FE code output
 * 
 * An interface to the \c DFT-FE code. [DFT-FE](https://sites.google.com/umich.edu/dftfe) 
 * is a C++ code for material modeling from first principles using Kohn-Sham density 
 * functional theory. It handles all-electron and pseudopotential calculations in the same 
 * framework while accommodating arbitrary boundary conditions. 
 */
class dftfe
{
public:
	/*!
	 * \brief Construct a new dftfe object
	 * 
	 */
	dftfe();

	/*!
	 * \brief Destroy the dftfe object
	 * 
	 */
	~dftfe();

	/*!
     * \brief Move constructor, construct a new dftfe object
     * 
     * \param other dftfe object
     */
	explicit dftfe(dftfe &&other);

	/*!
     * \brief Move assignment operator
     * 
     * \param other dftfe object
     * 
     * \returns dftfe& dftfe object
     */
	dftfe &operator=(dftfe &&other);

	/*!
	 * \brief reset the variables in dftfe
	 * 
	 */
	void reset();

	/*!
	 * \brief Set the Full File Name 
	 * 
	 * \param dataRootDirectory  Path to the root directory where data files are located
	 * \param baseFileName       Base part of the DFT-FE output files without number index
	 */
	inline void setFullFileName(std::string const &dataRootDirectory = "", std::string const &baseFileName = "");

	/*!
	 * \brief Get the Full File Name 
	 * 
	 * \returns std::string 
	 */
	inline std::string getFullFileName();

	/*!
	 * \brief Get the Total Number Run Files 
	 * 
	 * \returns std::size_t 
	 */
	std::size_t getTotalNumberRunFiles();

	/*!
	 * \brief Get the Total Number Steps 
	 * 
	 * \returns std::size_t 
	 */
	inline std::size_t getTotalNumberSteps();

	/*!
	 * \brief Get the Number Species
	 * 
	 * \returns std::size_t 
	 */
	inline std::size_t getNumberSpecies();

	/*!
	 * \brief Get the Number Species Types
	 * 
	 * \returns std::size_t 
	 */
	inline std::size_t getNumberSpeciesTypes();

	/*!
	 * \brief Get the Species attribute
	 * 
	 * \returns std::vector<umuq::speciesAttribute> 
	 */
	inline std::vector<umuq::speciesAttribute> getSpecies();

	/*!
	 * \brief Get the Species Information
	 * 
	 * \returns true 
	 * \returns false 
	 */
	bool getSpeciesInformation();

	/*!
	 * \brief Dump the system size and the coordinate of each species and their forces at each time step  
	 * 
	 * \param baseCoordinatesFileName  The name to be used as the default name for dumping each species coordinates
	 * \param baseForcesFileName       The name to be used as the default name for dumping each species forces
	 * 
	 * \returns true 
	 * \returns false 
	 */
	bool dump(std::string const &baseCoordinatesFileName = "COORDS", std::string const &baseForcesFileName = "FORCE");

private:
	/*!
     * \brief Delete a dftfe object copy construction
     * 
     * Avoiding implicit generation of the copy constructor.
     */
	dftfe(dftfe const &) = delete;

	/*!
     * \brief Delete a dftfe object assignment
     * 
     * Avoiding implicit copy assignment.
     */
	dftfe &operator=(dftfe const &) = delete;

private:
	/*! Full file name without suffix to read the data from the files */
	std::string fullFileName;

	/*! The starting index of the DFT-FE run files */
	std::size_t startIndexRunFiles;

	/*! Total number of run files from DFT-FE */
	std::size_t totalNumberRunFiles;

	/*! Total number of data points available in the DFT-FE input files */
	std::size_t totalNumberSteps;

	/*! Total number of species (atoms) */
	std::size_t nSpecies;

	/*! Species (atoms) types */
	std::size_t nSpeciesTypes;

	/*! Species with their attributes*/
	std::vector<umuq::speciesAttribute> Species;
};

dftfe::dftfe() : fullFileName(), startIndexRunFiles(0), totalNumberRunFiles(0), totalNumberSteps(0), nSpecies(0), nSpeciesTypes(0) {}

dftfe::~dftfe() {}

dftfe::dftfe(dftfe &&other)
{
	fullFileName = std::move(other.fullFileName);
	startIndexRunFiles = other.startIndexRunFiles;
	totalNumberRunFiles = other.totalNumberRunFiles;
	totalNumberSteps = other.totalNumberSteps;
	nSpecies = other.nSpecies;
	nSpeciesTypes = other.nSpeciesTypes;
	Species = std::move(other.Species);
}

dftfe &dftfe::operator=(dftfe &&other)
{
	fullFileName = std::move(other.fullFileName);
	startIndexRunFiles = other.startIndexRunFiles;
	totalNumberRunFiles = other.totalNumberRunFiles;
	totalNumberSteps = other.totalNumberSteps;
	nSpecies = other.nSpecies;
	nSpeciesTypes = other.nSpeciesTypes;
	Species = std::move(other.Species);

	return *this;
}

void dftfe::reset()
{
	fullFileName.clear();
	startIndexRunFiles = 0;
	totalNumberRunFiles = 0;
	totalNumberSteps = 0;
	nSpecies = 0;
	nSpeciesTypes = 0;
	Species.clear();
	Species.shrink_to_fit();
}

inline void dftfe::setFullFileName(std::string const &dataRootDirectory, std::string const &baseFileName)
{
	if (dataRootDirectory.size() || baseFileName.size())
	{
		fullFileName = dataRootDirectory + "/" + baseFileName + "%01d";
		return;
	}
	UMUQFAIL("No information to set the input file name!");
}

inline std::string dftfe::getFullFileName() { return fullFileName; }

std::size_t dftfe::getTotalNumberRunFiles()
{
	if (totalNumberRunFiles)
	{
		return totalNumberRunFiles;
	}
	if (fullFileName.size())
	{
		// Get an instance of the io object
		umuq::io file;

		// File name
		char fileName[LINESIZE];

		// Initialize the starting index
		startIndexRunFiles = 0;

		sprintf(fileName, fullFileName.c_str(), startIndexRunFiles++);
		for (;;)
		{
			if (file.isFileExist(fileName))
			{
				startIndexRunFiles--;
				break;
			}
			sprintf(fileName, fullFileName.c_str(), startIndexRunFiles++);
			if (startIndexRunFiles > umuq::HugeCost)
			{
				UMUQFAILRETURN("The starting index of the input file = ", startIndexRunFiles, " is out of bound!");
			}
		}

		// Loop through the files and count number of steps
		for (std::size_t fileID = startIndexRunFiles;; fileID++)
		{
			sprintf(fileName, fullFileName.c_str(), fileID);
			if (file.isFileExist(fileName))
			{
				totalNumberRunFiles++;
				continue;
			}
			break;
		}
	}
	return totalNumberRunFiles;
}

inline std::size_t dftfe::getTotalNumberSteps() { return totalNumberSteps; }

inline std::size_t dftfe::getNumberSpecies() { return nSpecies; }

inline std::size_t dftfe::getNumberSpeciesTypes() { return nSpeciesTypes; }

inline std::vector<umuq::speciesAttribute> dftfe::getSpecies() { return Species; }

bool dftfe::getSpeciesInformation()
{
	if (!getTotalNumberRunFiles())
	{
		if (fullFileName.size())
		{
			UMUQFAILRETURN("There is no input file to open!");
		}
		UMUQFAILRETURN("Please set the file name!");
	}
	{
		// Get an instance of the io object
		umuq::io file;

		// File name
		char fileName[LINESIZE];

		sprintf(fileName, fullFileName.c_str(), startIndexRunFiles);
		if (file.openFile(fileName, file.in))
		{
			// Get an instance of a parser object to parse
			umuq::parser p;

			nSpecies = 0;
			nSpeciesTypes = 0;

			// Read each line in the file and skip all the commented and empty line with the default comment "#"
			while (file.readLine())
			{
				// Parse the line into line arguments
				p.parse(file.getLine());
				if (p.at<std::string>(0) == "set")
				{
					if (p.at<std::string>(1) == "NATOMS")
					{
						nSpecies = p.at<int>(3);
					}
					if (p.at<std::string>(1) == "NATOM")
					{
						if (p.at<std::string>(2) == "TYPES")
						{
							nSpeciesTypes = p.at<int>(4);
						}
					}
					if (nSpecies && nSpeciesTypes)
					{
						break;
					}
				}
			}

			{
				// Create an instance of the species object
				umuq::species s;

				// Read each line in the file and skip all the commented and empty line with the default comment "#"
				while (file.readLine())
				{
					// Parse the line into line arguments
					p.parse(file.getLine());
					if (p.at<std::string>(0).substr(0, 2) == "Z:")
					{
						auto index = p.at<int>(1);
						Species.push_back(s.getSpecies(index));
						if (Species.size() == nSpeciesTypes)
						{
							break;
						}
					}
				}

				if (!Species.size())
				{
					file.rewindFile();

					// Read each line in the file and skip all the commented and empty line with the default comment "#"
					while (file.readLine())
					{
						// Parse the line into line arguments
						p.parse(file.getLine());
						if (p.at<std::string>(0) == "Number")
						{
							if (p.getLineArgNum() > 5)
							{
								if (p.at<std::string>(6).substr(0, 2) == "Z:")
								{
									auto index = p.at<int>(7);
									Species.push_back(s.getSpecies(index));
									if (Species.size() == nSpeciesTypes)
									{
										break;
									}
								}
							}
						}
					}
				}
			}

			file.closeFile();

			totalNumberSteps = 0;

			// Loop through the files and count number of steps
			for (std::size_t fileID = startIndexRunFiles; fileID < startIndexRunFiles + getTotalNumberRunFiles(); fileID++)
			{
				sprintf(fileName, fullFileName.c_str(), fileID);
				if (file.isFileExist(fileName))
				{
					if (file.openFile(fileName, file.in))
					{
						// Read each line in the file and skip all the commented and empty line with the default comment "#"
						while (file.readLine())
						{
							// Parse the line into line arguments
							p.parse(file.getLine());
							if (p.at<std::string>(0) == "v1")
							{
								totalNumberSteps++;
							}
						}
						file.closeFile();
					}
					continue;
				}
				return false;
			}
			return true;
		}
		UMUQFAILRETURN("There is no input file to open!");
	}
}

bool dftfe::dump(std::string const &baseCoordinatesFileName, std::string const &baseForcesFileName)
{
	if (!totalNumberSteps)
	{
		if (!getSpeciesInformation())
		{
			UMUQFAILRETURN("Failed to read the information from input files!")
		}
	}
	{
		// Get an instance of the io object
		umuq::io file;

		// File name
		char fileName[LINESIZE];

		// Get an instance of the units class with METAL style
		umuq::units mdUnits("METAL");

		// Set the unit style for conversion from DFT-FE ELECTRON style to METAL
		if (!mdUnits.convertFromStyle("ELECTRON"))
		{
			UMUQFAILRETURN("Can not set the conversion factors!");
		}

		// Unit cell coordinates
		std::vector<double> unitCells(9);

		// Species coordinates
		std::vector<double> fractionalCoordinates(nSpecies * 3);

		// Species force
		std::vector<double> force(nSpecies * 3);

		std::size_t numberSteps = 0;

		for (std::size_t fileID = startIndexRunFiles; fileID < startIndexRunFiles + getTotalNumberRunFiles(); fileID++)
		{
			sprintf(fileName, fullFileName.c_str(), fileID);
			if (file.openFile(fileName, file.in))
			{
				// An instance of a parser object to parse
				umuq::parser p;

				std::size_t Id = 0;
				std::size_t dumpFlag = 0;

				while (file.readLine())
				{
					if (dumpFlag == 2)
					{
						// Get an instance of the io object
						umuq::io outputFile;

						std::string FileName = baseCoordinatesFileName + "%01d";

						sprintf(fileName, FileName.c_str(), numberSteps - 1);
						if (outputFile.openFile(fileName, outputFile.out))
						{
							// Write the matrix in it
							outputFile.saveMatrix<double>(unitCells, 3, 3);
							outputFile.closeFile();
						}

						FileName = "X" + baseCoordinatesFileName + "%01d";

						sprintf(fileName, FileName.c_str(), numberSteps - 1);
						if (outputFile.openFile(fileName, outputFile.out))
						{
							// Write the matrix in it
							outputFile.saveMatrix<double>(fractionalCoordinates, nSpecies, 3);
							outputFile.closeFile();
						}

						FileName = baseForcesFileName + "%01d";

						sprintf(fileName, FileName.c_str(), numberSteps - 1);
						if (outputFile.openFile(fileName, outputFile.out))
						{
							// Write the matrix in it
							outputFile.saveMatrix<double>(force, nSpecies, 3);
							outputFile.closeFile();
						}

						dumpFlag = 0;
					}

					// Parse the line into line arguments
					p.parse(file.getLine());
					if (p.at<std::string>(0) == "v1")
					{
						unitCells[0] = p.at<double>(1);
						unitCells[1] = p.at<double>(2);
						unitCells[2] = p.at<double>(3);
						if (file.readLine())
						{
							// Parse the line into line arguments
							p.parse(file.getLine());
							unitCells[3] = p.at<double>(1);
							unitCells[4] = p.at<double>(2);
							unitCells[5] = p.at<double>(3);
							if (file.readLine())
							{
								// Parse the line into line arguments
								p.parse(file.getLine());
								unitCells[6] = p.at<double>(1);
								unitCells[7] = p.at<double>(2);
								unitCells[8] = p.at<double>(3);
							}
						}

						// Convert from DFT-FE style to the output METAL style
						mdUnits.convertLength(unitCells);

						numberSteps++;
						dumpFlag++;
					}

					if (p.at<std::string>(0) == "AtomId")
					{
						if (Id)
						{
							Id = 0;
							force[Id++] = p.at<double>(2);
							force[Id++] = p.at<double>(3);
							force[Id++] = p.at<double>(4);
							while (file.readLine())
							{
								// Parse the line into line arguments
								p.parse(file.getLine());
								if (p.at<std::string>(0) == "AtomId")
								{
									force[Id++] = p.at<double>(2);
									force[Id++] = p.at<double>(3);
									force[Id++] = p.at<double>(4);
									continue;
								}
								break;
							}
#ifdef DEBUG
							if (Id / 3 != nSpecies)
							{
								UMUQFAILRETURN("There is a mismatch in the number of force data points = ", Id / 3, "!= ", nSpecies, " !");
							}
#endif

							// Convert from DFT-FE style to the output METAL style
							mdUnits.convertForce(force);
							dumpFlag++;

							continue;
						}
						else
						{
							fractionalCoordinates[Id++] = p.at<double>(2);
							fractionalCoordinates[Id++] = p.at<double>(3);
							fractionalCoordinates[Id++] = p.at<double>(4);
							while (file.readLine())
							{
								// Parse the line into line arguments
								p.parse(file.getLine());
								if (p.at<std::string>(0) == "AtomId")
								{
									fractionalCoordinates[Id++] = p.at<double>(2);
									fractionalCoordinates[Id++] = p.at<double>(3);
									fractionalCoordinates[Id++] = p.at<double>(4);
									continue;
								}
								break;
							}
#ifdef DEBUG
							if (Id / 3 != nSpecies)
							{
								UMUQFAILRETURN("There is a mismatch in the number of coordinates data points = ", Id / 3, "!= ", nSpecies, " !");
							}
#endif
							// Convert from DFT-FE style to the output style
							mdUnits.convertLength(fractionalCoordinates);
							dumpFlag++;

							// Convert and correct the species coordinates
							umuq::convertFractionalToCartesianCoordinates(unitCells, fractionalCoordinates);

							continue;
						}
					}
				}
				file.closeFile();

				continue;
			}
			UMUQFAILRETURN("Failed to open the file!")
		}
		if (numberSteps != totalNumberSteps)
		{
			UMUQFAILRETURN("The read number of steps = ", numberSteps, "!= ", totalNumberSteps, " !");
		}
		return true;
	}
}

} // namespace umuq

#endif // UMUQ_DFTFE