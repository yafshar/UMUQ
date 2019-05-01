#ifndef UMUQ_DFTFE_H
#define UMUQ_DFTFE_H

#include "core/core.hpp"
#include "io/io.hpp"
#include "misc/parser.hpp"
#include "misc/utility.hpp"
#include "units/speciesname.hpp"
#include "units/units.hpp"

namespace umuq
{

/*!
 * \defgroup 3rdparty_Module 3rdparty module
 *
 * This is the 3rdparty Module of UMUQ providing all necessary classes
 * for an interface to the 3rd party codes like DFT-FE, LAMMPS, ...
 * currently supported in UMUQ.
 */

/*! \class dftfe
 * \ingroup 3rdparty_Module
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
     * \param dataDirectory  Path to the directory where data files are located
     * \param baseFileName   Base part of the DFT-FE output run files without number index
     * \param paramFileName  Parameters file name used by \c DFT-FE
     */
    inline void setFullFileName(std::string const &dataDirectory = "", std::string const &baseFileName = "", std::string const &paramFileName = "");

    /*!
     * \brief Get the Full File Name
     *
     * \returns std::string
     */
    inline std::string getFullFileName();

    /*!
     * \brief Get the Parameter File Name object
     *
     * \returns std::string
     */
    inline std::string getParameterFileName();

    /*!
     * \brief Get the Total Number Run Files exist on the PATH
     *
     * \returns std::size_t
     *
     * \note
     * This function search the path which is set by the \b setFullFileName function \sa umuq::dftfe::setFullFileName
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
     * \brief Get the Species Types object
     *
     * \returns std::vector<int>
     */
    inline std::vector<int> getSpeciesTypes();

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
     * \param format                   The trajectory file output format
     *
     * \returns true
     * \returns false
     */
    bool dump(std::string const &baseCoordinatesFileName = "COORDS", std::string const &baseForcesFileName = "FORCE", std::string const &format = "");

    /*!
     * \brief To calculate mean square displacement for each species
     *
     * \return true
     * \return false
     */
    bool calculateMeanSquareDisplacement();

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
    /*! Data root directory*/
    std::string dataRootDirectory;

    /*! Full file name without suffix to read the data from the files */
    std::string fullFileName;

    /*! Full parameter file name */
    std::string parameterFileName;

    /*! The starting index of the DFT-FE run files */
    std::size_t startIndexRunFiles;

    /*! Total number of run files from DFT-FE */
    std::size_t totalNumberRunFiles;

    /*! Total number of data points available in the DFT-FE input files */
    std::size_t totalNumberSteps;

    /*! Total number of species (atoms) */
    std::size_t nSpecies;

    /*! Number of species (atoms) types */
    std::size_t nSpeciesTypes;

    /*! Species with their attributes*/
    std::vector<umuq::speciesAttribute> Species;

    /*! Species (atoms) types */
    std::vector<int> speciesTypes;

    /*! Species mean square displacement */
    std::vector<double> msd;
};

dftfe::dftfe() : dataRootDirectory(), fullFileName(), parameterFileName(), startIndexRunFiles(0), totalNumberRunFiles(0), totalNumberSteps(0), nSpecies(0), nSpeciesTypes(0) {}

dftfe::~dftfe() {}

dftfe::dftfe(dftfe &&other)
{
    dataRootDirectory = std::move(other.dataRootDirectory);
    fullFileName = std::move(other.fullFileName);
    parameterFileName = std::move(other.parameterFileName);
    startIndexRunFiles = other.startIndexRunFiles;
    totalNumberRunFiles = other.totalNumberRunFiles;
    totalNumberSteps = other.totalNumberSteps;
    nSpecies = other.nSpecies;
    nSpeciesTypes = other.nSpeciesTypes;
    Species = std::move(other.Species);
    speciesTypes = std::move(other.speciesTypes);
    msd = std::move(other.msd);
}

dftfe &dftfe::operator=(dftfe &&other)
{
    dataRootDirectory = std::move(other.dataRootDirectory);
    fullFileName = std::move(other.fullFileName);
    parameterFileName = std::move(other.parameterFileName);
    startIndexRunFiles = other.startIndexRunFiles;
    totalNumberRunFiles = other.totalNumberRunFiles;
    totalNumberSteps = other.totalNumberSteps;
    nSpecies = other.nSpecies;
    nSpeciesTypes = other.nSpeciesTypes;
    Species = std::move(other.Species);
    speciesTypes = std::move(other.speciesTypes);
    msd = std::move(other.msd);

    return *this;
}

void dftfe::reset()
{
    dataRootDirectory.clear();
    fullFileName.clear();
    parameterFileName.clear();
    startIndexRunFiles = 0;
    totalNumberRunFiles = 0;
    totalNumberSteps = 0;
    nSpecies = 0;
    nSpeciesTypes = 0;
    Species.clear();
    Species.shrink_to_fit();
    speciesTypes.clear();
    speciesTypes.shrink_to_fit();
    msd.clear();
    msd.shrink_to_fit();
}

inline void dftfe::setFullFileName(std::string const &dataDirectory, std::string const &baseFileName, std::string const &paramFileName)
{
    umuq::utility u;
    dataRootDirectory = (dataDirectory.size() ? dataDirectory : u.getCurrentWorkingDirectory()) + "/";
    if (baseFileName.size())
    {
        fullFileName = dataRootDirectory + baseFileName + "%01d";
        parameterFileName = dataRootDirectory + (paramFileName.size() ? paramFileName : "parameterFile.prm");
        return;
    }
    UMUQFAIL("No information to set the input file name!");
}

inline std::string dftfe::getFullFileName() { return fullFileName; }

inline std::string dftfe::getParameterFileName() { return parameterFileName; }

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

inline std::vector<int> dftfe::getSpeciesTypes() { return speciesTypes; }

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
        if (file.openFile(fileName))
        {
            // Get an instance of a parser object to parse
            umuq::parser p;
            // counter for the number of species
            nSpecies = 0;
            // Counter for the number of species types
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

            // Set the total number of steps to zero
            totalNumberSteps = 0;

            // Loop through the files and count number of steps
            for (std::size_t fileID = startIndexRunFiles; fileID < startIndexRunFiles + getTotalNumberRunFiles(); fileID++)
            {
                sprintf(fileName, fullFileName.c_str(), fileID);
                if (file.isFileExist(fileName))
                {
                    if (file.openFile(fileName))
                    {
                        std::size_t stepFlag = 0;
                        // Read each line in the file and skip all the commented and empty line with the default comment "#"
                        while (file.readLine())
                        {
                            if (stepFlag == 3)
                            {
                                totalNumberSteps++;
                                stepFlag = 0;
                            }
                            // Parse the line into line arguments
                            p.parse(file.getLine());
                            if (p.at<std::string>(0) == "v1")
                            {
                                stepFlag = 1;
                                continue;
                            }
                            if (p.at<std::string>(0) == "AtomId")
                            {
                                auto Id = 1;
                                while (file.readLine())
                                {
                                    // Parse the line into line arguments
                                    p.parse(file.getLine());
                                    if (p.at<std::string>(0) == "AtomId")
                                    {
                                        Id++;
                                    }
                                    if (Id == nSpecies)
                                    {
                                        break;
                                    }
                                }
                                stepFlag++;
                                continue;
                            }
                        }
                        file.closeFile();
                    }
                    continue;
                }
                return false;
            }
            if (nSpeciesTypes > 1)
            {
                if (file.isFileExist(parameterFileName))
                {
                    if (file.openFile(parameterFileName))
                    {
                        // Read each line in the file and skip all the commented and empty line with the default comment "#"
                        while (file.readLine())
                        {
                            // Parse the line into line arguments
                            p.parse(file.getLine());
                            if (p.at<std::string>(0) == "set")
                            {
                                if (p.at<std::string>(1) == "ATOMIC")
                                {
                                    auto str = file.getLine();
                                    p.parse(str.replace(str.find_first_of("="), 1, " "));
                                    sprintf(fileName, "%s%s", dataRootDirectory.c_str(), p.at<std::string>(4).c_str());
                                    break;
                                }
                            }
                        }
                        file.closeFile();
                        if (file.isFileExist(fileName))
                        {
                            if (file.openFile(fileName))
                            {
                                // Create an instance of the species object
                                umuq::species s;

                                // Read each line in the file and skip all the commented and empty line with the default comment "#"
                                while (file.readLine())
                                {
                                    // Parse the line into line arguments
                                    p.parse(file.getLine());
                                    auto index = p.at<int>(0);
                                    auto sp = s.getSpecies(index);
                                    for (auto i = 0; i < nSpeciesTypes; i++)
                                    {
                                        if (sp.name == Species[i].name)
                                        {
                                            speciesTypes.push_back(i);
                                            break;
                                        }
                                    }
                                }
                                file.closeFile();
                                return (speciesTypes.size() == nSpecies);
                            }
                        }
                    }
                }
                return false;
            }
            else
            {
                speciesTypes.resize(nSpecies, 0);
            }
            return true;
        }
        UMUQFAILRETURN("There is no input file to open!");
    }
}

bool dftfe::dump(std::string const &baseCoordinatesFileName, std::string const &baseForcesFileName, std::string const &format)
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

        // Get an instance of the io object
        umuq::io outputFile;

        // Get an instance of the io object
        umuq::io xyzFile;

        // Create flags to indicate the requested tasks
        bool const coordsFileFlag = baseCoordinatesFileName.size();
        bool const forceFileFlag = baseForcesFileName.size();
        bool const xyzFileFlag = (format == "xyz") ? xyzFile.openFile("history.xyz", xyzFile.out) : false;

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
        std::vector<double> boundingVectors(9);

        // Species coordinates
        std::vector<double> fractionalCoordinates(nSpecies * 3);

        // Species force
        std::vector<double> force(nSpecies * 3);

        // Number of steps counter
        std::size_t numberSteps = 0;

        // Loop through all the files
        for (std::size_t fileID = startIndexRunFiles; fileID < startIndexRunFiles + getTotalNumberRunFiles(); fileID++)
        {
            sprintf(fileName, fullFileName.c_str(), fileID);
            if (file.openFile(fileName))
            {
                // An instance of a parser object to parse
                umuq::parser p;

                // Set an index indicator for coordinates and forces
                std::size_t Id = 0;

                // Set the dumping flag
                std::size_t dumpFlag = 0;

                while (file.readLine())
                {
                    if (dumpFlag == 3)
                    {
                        if (coordsFileFlag)
                        {
                            std::string FileName = baseCoordinatesFileName + "%01d";
                            sprintf(fileName, FileName.c_str(), numberSteps);
                            if (outputFile.openFile(fileName, outputFile.out))
                            {
                                // Write the matrix in it
                                outputFile.saveMatrix<double>(boundingVectors, 3, 3);
                                outputFile.closeFile();
                            }

                            FileName = "X" + baseCoordinatesFileName + "%01d";

                            sprintf(fileName, FileName.c_str(), numberSteps);
                            if (outputFile.openFile(fileName, outputFile.out))
                            {
                                // Write the matrix in it
                                outputFile.saveMatrix<double>(fractionalCoordinates, nSpecies, 3);
                                outputFile.closeFile();
                            }
                        }

                        if (forceFileFlag)
                        {
                            std::string FileName = baseForcesFileName + "%01d";
                            sprintf(fileName, FileName.c_str(), numberSteps);
                            if (outputFile.openFile(fileName, outputFile.out))
                            {
                                // Write the matrix in it
                                outputFile.saveMatrix<double>(force, nSpecies, 3);
                                outputFile.closeFile();
                            }
                        }

                        if (xyzFileFlag)
                        {
                            // Get the IO file base stream
                            auto &fs = xyzFile.getFstream();
                            fs << nSpecies << "\n";
                            // fs << "step\t" << numberSteps << "\n";
                            fs << "Lattice=\"";
                            for (auto c : boundingVectors)
                            {
                                fs << " " << c;
                            }
                            fs << "\" Properties=species:S:1:pos:R:3\n";
                            auto coordsIt = fractionalCoordinates.begin();
                            for (auto speciesIndex = 0; speciesIndex < nSpecies; speciesIndex++)
                            {
                                fs << Species[speciesTypes[speciesIndex]].name << "\t";
                                fs << *coordsIt++ << " ";
                                fs << *coordsIt++ << " ";
                                fs << *coordsIt++ << "\n";
                            }
                        }

                        numberSteps++;
                        dumpFlag = 0;
                    }

                    // Parse the line into line arguments
                    p.parse(file.getLine());
                    if (p.at<std::string>(0) == "v1")
                    {
                        boundingVectors[0] = p.at<double>(1);
                        boundingVectors[1] = p.at<double>(2);
                        boundingVectors[2] = p.at<double>(3);
                        if (file.readLine())
                        {
                            // Parse the line into line arguments
                            p.parse(file.getLine());
                            boundingVectors[3] = p.at<double>(1);
                            boundingVectors[4] = p.at<double>(2);
                            boundingVectors[5] = p.at<double>(3);
                            if (file.readLine())
                            {
                                // Parse the line into line arguments
                                p.parse(file.getLine());
                                boundingVectors[6] = p.at<double>(1);
                                boundingVectors[7] = p.at<double>(2);
                                boundingVectors[8] = p.at<double>(3);
                            }
                        }

                        // Convert from DFT-FE style to the output METAL style
                        mdUnits.convertLength(boundingVectors);

                        dumpFlag = 1;
                        Id = 0;
                        continue;
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
                                if (Id / 3 == nSpecies)
                                {
                                    break;
                                }
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
                                if (Id / 3 == nSpecies)
                                {
                                    break;
                                }
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

                            // Convert the fractional coordinates to the Cartesian coordinates
                            umuq::convertFractionalToCartesianCoordinates(boundingVectors, fractionalCoordinates);

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
        if (xyzFileFlag)
        {
            xyzFile.closeFile();
        }
        return true;
    }
}

bool dftfe::calculateMeanSquareDisplacement()
{
    if (!totalNumberSteps)
    {
        if (!getSpeciesInformation())
        {
            UMUQFAILRETURN("Failed to read the information from input files!")
        }
    }

    std::size_t index = 0;
    std::size_t ltype = 0;

    // Initialize msd arrays
    std::size_t lsr = 0;
    std::size_t msr = 0;
    std::size_t nsmsd = 0;

    // Unit cell coordinates
    std::vector<double> boundingVectors(9);

    // Species coordinates
    std::vector<double> inCoordinates(nSpecies * 3, 0);
    std::vector<double> toCoordinates(nSpecies * 3, 0);
    std::vector<double> diffCoordinates(nSpecies * 3);
    std::vector<double> msd0(nSpecies * 3 * totalNumberSteps, 0);
    std::vector<double> msd(nSpecies * totalNumberSteps, 0);
    std::vector<double> rr2(totalNumberSteps, 0);

    std::vector<int> msm(totalNumberSteps, 0);
    std::vector<int> imd(totalNumberSteps, 0);

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

        // Number of steps counter
        std::size_t numberSteps = 0;

        // Loop through all the files
        for (std::size_t fileID = startIndexRunFiles; fileID < startIndexRunFiles + getTotalNumberRunFiles(); fileID++)
        {
            sprintf(fileName, fullFileName.c_str(), fileID);
            if (file.openFile(fileName))
            {
                // An instance of a parser object to parse
                umuq::parser p;

                // Set an index indicator for coordinates and forces
                std::size_t Id = 0;

                // Set the dumping flag
                std::size_t dumpFlag = 0;

                while (file.readLine())
                {
                    if (dumpFlag == 3)
                    {
                        if (numberSteps)
                        {
                            auto toIt = toCoordinates.begin();
                            auto inIt = inCoordinates.begin();
                            // Calculate mean square displacement
                            std::for_each(diffCoordinates.begin(), diffCoordinates.end(), [&](auto &diff) { diff = *toIt++ - *inIt++; });
                        }

                        std::copy(toCoordinates.begin(), toCoordinates.end(), inCoordinates.begin());

                        numberSteps++;
                        dumpFlag = 0;
                    }

                    // Parse the line into line arguments
                    p.parse(file.getLine());
                    if (p.at<std::string>(0) == "v1")
                    {
                        boundingVectors[0] = p.at<double>(1);
                        boundingVectors[1] = p.at<double>(2);
                        boundingVectors[2] = p.at<double>(3);
                        if (file.readLine())
                        {
                            // Parse the line into line arguments
                            p.parse(file.getLine());
                            boundingVectors[3] = p.at<double>(1);
                            boundingVectors[4] = p.at<double>(2);
                            boundingVectors[5] = p.at<double>(3);
                            if (file.readLine())
                            {
                                // Parse the line into line arguments
                                p.parse(file.getLine());
                                boundingVectors[6] = p.at<double>(1);
                                boundingVectors[7] = p.at<double>(2);
                                boundingVectors[8] = p.at<double>(3);
                            }
                        }

                        // Convert from DFT-FE style to the output METAL style
                        mdUnits.convertLength(boundingVectors);

                        dumpFlag = 1;
                        Id = 0;
                        continue;
                    }
                    if (p.at<std::string>(0) == "AtomId")
                    {
                        if (Id)
                        {
                            Id = 1;
                            while (file.readLine())
                            {
                                // Parse the line into line arguments
                                p.parse(file.getLine());
                                if (p.at<std::string>(0) == "AtomId")
                                {
                                    Id++;
                                    continue;
                                }
                                if (Id == nSpecies)
                                {
                                    break;
                                }
                            }
                            dumpFlag++;
                            continue;
                        }
                        else
                        {
                            toCoordinates[Id++] = p.at<double>(2);
                            toCoordinates[Id++] = p.at<double>(3);
                            toCoordinates[Id++] = p.at<double>(4);
                            while (file.readLine())
                            {
                                // Parse the line into line arguments
                                p.parse(file.getLine());
                                if (p.at<std::string>(0) == "AtomId")
                                {
                                    toCoordinates[Id++] = p.at<double>(2);
                                    toCoordinates[Id++] = p.at<double>(3);
                                    toCoordinates[Id++] = p.at<double>(4);
                                    continue;
                                }
                                if (Id / 3 == nSpecies)
                                {
                                    break;
                                }
                            }
                            // Convert from DFT-FE style to the output style
                            mdUnits.convertLength(toCoordinates);
                            dumpFlag++;
                            // Convert the fractional coordinates to the Cartesian coordinates
                            umuq::convertFractionalToCartesianCoordinates(boundingVectors, toCoordinates);
                            continue;
                        }
                    }
                }
                file.closeFile();
            }
        }
    }
}

} // namespace umuq

#endif // UMUQ_DFTFE