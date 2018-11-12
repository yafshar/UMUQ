#include "core/core.hpp"
#include "io/io.hpp"
#include "misc/parser.hpp"
#include "gtest/gtest.h"

// TEST for file existence
TEST(isFileExist_test, HandlesFiles)
{
    umuq::io f;
    EXPECT_TRUE(f.isFileExist("./data/test.txt"));
    EXPECT_FALSE(f.isFileExist("utility.txt"));
}

// TEST how IO handles files
TEST(openFile_test, HandlesFiles)
{
    // An instance of io class
    umuq::io f;

    EXPECT_FALSE(f.isFileOpened());

    EXPECT_TRUE(f.openFile("./data/test.txt"));

    EXPECT_TRUE(f.isFileOpened());
    
    int n = 0;
    while (f.readLine())
    {
        // count the number of non empty and not commented line with "#" as default comment
        n++;
    }
    
    EXPECT_EQ(n, 26);
    
    f.closeFile();
    
    EXPECT_FALSE(f.isFileOpened());
}

// TEST how IO handles std::fstream
TEST(openFilestream_test, HandlesFiles)
{
    const char *fileName = "iotmp";
    std::remove(fileName);

    // An instance of io class
    umuq::io f;

    EXPECT_FALSE(f.openFile(fileName, f.in));
    EXPECT_FALSE(f.openFile(fileName, f.in | f.binary));
    EXPECT_TRUE(f.openFile(fileName, f.out));

    std::fstream &fs = f.getFstream();

    if (fs.is_open())
    {
        f.closeFile();
    }

    std::remove(fileName);
}

/*! 
 * Load and Save of an array with a matrix format
 */
TEST(io_test, HandlesLoadandSaveArray)
{
    const char *fileName = "iotmp";

    // An instance of io class
    umuq::io f;

    // - 1

    // Create a new array and initialize it
    int *E = new int[12];
    for (int i = 0; i < 12; i++)
    {
        E[i] = i;
    }

    // Create a new array
    int *F = new int[12];

    // Open a file for reading and writing
    if (f.openFile(fileName, f.in | f.out | f.trunc))
    {
        // Save the array in a matrix format
        f.saveMatrix<int>(E, 3, 4);

        // Rewind the file
        f.rewindFile();

        // Read the array
        f.loadMatrix<int>(F, 3, 4);

        for (int i = 0, l = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++, l++)
            {
                EXPECT_EQ(F[l], E[l]);
            }
        }
        f.closeFile();
    }

    delete[] F;

    // Create a new array
    F = new int[24];

    // Open a file for reading and writing
    if (f.openFile(fileName, f.in | f.out | f.trunc))
    {
        // Save the array in a vector format and keep the stream pointer at the end of line
        f.saveMatrix<int>(E, 12, 1, 2);
        f.saveMatrix<int>(E, 12);

        // Rewind the file
        f.rewindFile();

        // Read the array
        f.loadMatrix<int>(F, 24);

        for (int i = 0, l = 0; i < 24; i++, l++)
        {
            if (i == 12)
                l = 0;
            EXPECT_EQ(F[i], E[l]);
        }

        f.closeFile();

        // delete the file
        std::remove(fileName);
    }

    delete[] E;
    delete[] F;
}

/*! 
 * Load and Save array of pointers in a matrix format from and to a file 
 */
TEST(io_test, HandlesLoadandSaveArrayofPointers)
{
    const char *fileName = "iotmp";

    // An instance of io class
    umuq::io f;

    // - 2

    // Create a new array and initialize it
    double **G = nullptr;
    G = new double *[3];
    for (int i = 0; i < 3; i++)
    {
        G[i] = new double[4];
    }
    for (int i = 0, l = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++, l++)
        {
            G[i][j] = (double)l;
        }
    }

    // Create a new array
    double H[3][4];

    // Open a file for reading and writing
    if (f.openFile(fileName, f.in | f.out | f.trunc))
    {
        // Write the matrix
        f.saveMatrix<double>(G, 3, 4);

        // Rewind the file
        f.rewindFile();

        // Read the matrix
        f.loadMatrix<double>(reinterpret_cast<double *>(H), 3, 4);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                EXPECT_DOUBLE_EQ(H[i][j], G[i][j]);
            }
        }

        f.closeFile();

        // delete the file
        std::remove(fileName);
    }

    delete[] * G;
    delete[] G;
}

/*! 
 * Load and Save of two different data types
 */
TEST(io_test, HandlesLoadandSaveDifferentData)
{
    const char *fileName = "iotmp";

    // An instance of io class
    umuq::io f;

    // - 3

    // Create a new array and initialize it
    double **K = nullptr;
    K = new double *[3];
    for (int i = 0; i < 3; i++)
    {
        K[i] = new double[8];
    }
    for (int i = 0, l = 0; i < 3; i++)
    {
        for (int j = 0; j < 8; j++, l++)
        {
            K[i][j] = (double)l;
        }
    }

    // Create a new array and initialize it
    int *L = new int[20];
    for (int i = 0; i < 20; i++)
    {
        L[i] = i;
    }

    // Open a file for reading and writing
    if (f.openFile(fileName, f.in | f.out | f.trunc))
    {

        // Write the matrices
        f.saveMatrix<double>(K, 3, 8);
        f.saveMatrix<int>(L, 20);

        // Rewind the file
        f.rewindFile();

        double M[3][8];
        int N[20];

        f.loadMatrix<double>(reinterpret_cast<double *>(M), 3, 8);
        f.loadMatrix<int>(N, 20);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                EXPECT_DOUBLE_EQ(M[i][j], K[i][j]);
            }
        }

        for (int i = 0; i < 20; i++)
        {
            EXPECT_EQ(N[i], L[i]);
        }

        f.closeFile();
        // delete the file
        std::remove(fileName);
    }

    delete[] * K;
    delete[] K;
    delete[] L;
}

/*! 
 * Load and Save of an array of pointers from and to a file 
 */
TEST(io_test, HandlesLoadandSaveDoubleArrays)
{

    const char *fileName = "iotmp";

    // An instance of io class
    umuq::io f;

    // - 4

    // Create a new array and initialize it
    double **K = nullptr;
    K = new double *[3];
    for (int i = 0; i < 3; i++)
    {
        K[i] = new double[6];
    }
    for (int i = 0, l = 0; i < 3; i++)
    {
        for (int j = 0; j < 6; j++, l++)
        {
            K[i][j] = (double)l;
        }
    }

    double **M = nullptr;
    M = new double *[3];
    for (int i = 0; i < 3; i++)
    {
        M[i] = new double[6];
    }

    // Open a file for reading and writing
    if (f.openFile(fileName, f.in | f.out | f.trunc))
    {
        // Write the matrices
        f.saveMatrix<double>(K, 3, 6);

        // Rewind the file
        f.rewindFile();

        f.loadMatrix<double>(M, 3, 6);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                EXPECT_DOUBLE_EQ(M[i][j], K[i][j]);
            }
        }

        f.closeFile();
        // delete the file
        std::remove(fileName);
    }

    delete[] * K;
    delete[] K;
    delete[] * M;
    delete[] M;
}

/*! 
 * Load and Save of DataStructure from and to a file 
 */
TEST(io_test, HandlesLoadandSaveDataStructure)
{

    const char *fileName = "iotmp";

    // An instance of io class
    umuq::io f;

    // - 5

    struct ebasic
    {
        double *Parray;
        int ndimParray;
        double *Garray;
        int ndimGarray;
        double Fvalue;
        int surrogate;
        int nsel;
        /*!
             *  \brief constructor for the default variables
             *
             */
        ebasic() : Parray(NULL),
                   ndimParray(0),
                   Garray(NULL),
                   ndimGarray(0),
                   Fvalue(0),
                   surrogate(0),
                   nsel(0){};
    };

    class edatabase
    {
      public:
        ebasic *entry;
        int entries;
        edatabase() : entry(NULL),
                      entries(0){};
        ~edatabase()
        {
            destroy();
        }

      private:
        void destroy()
        {
            for (int i = 0; i < entries; i++)
            {
                if (entry[i].Parray != NULL)
                {
                    delete[] entry[i].Parray;
                }
                if (entry[i].Garray != NULL)
                {
                    delete[] entry[i].Garray;
                }
            }
            if (entry != NULL)
            {
                delete[] entry;
            }
        }
    };

    // Create data and initialize it
    edatabase dd;

    dd.entries = 4;
    dd.entry = new ebasic[dd.entries];
    for (int i = 0, l = 0; i < dd.entries; i++)
    {
        dd.entry[i].ndimParray = 2;
        dd.entry[i].Parray = new double[dd.entry[i].ndimParray];
        for (int j = 0; j < dd.entry[i].ndimParray; j++)
        {
            l++;
            dd.entry[i].Parray[j] = (double)(l);
        }

        dd.entry[i].ndimGarray = 4;
        dd.entry[i].Garray = new double[dd.entry[i].ndimGarray];
        for (int j = 0; j < dd.entry[i].ndimGarray; j++)
        {
            l++;
            dd.entry[i].Garray[j] = (double)(l);
        }
        l++;
        dd.entry[i].Fvalue = (double)(l) + 1000.;
    }

    // Open a file for reading and writing
    if (f.openFile(fileName, f.in | f.out | f.trunc))
    {

        double **tmp = nullptr;
        tmp = new double *[2];

        for (int i = 0; i < dd.entries; i++)
        {
            tmp[0] = dd.entry[i].Parray;
            tmp[1] = &dd.entry[i].Fvalue;

            int nCols[2];
            nCols[0] = dd.entry[i].ndimParray;
            nCols[1] = 1;

            f.saveMatrix<double>(tmp, 2, nCols, 2);
            f.saveMatrix<double>(dd.entry[i].Garray, 1, dd.entry[i].ndimGarray);
        }

        // Rewind the file
        f.rewindFile();

        // Create data and initialize it
        edatabase ee;
        ee.entries = 4;
        ee.entry = new ebasic[ee.entries];
        for (int i = 0; i < ee.entries; i++)
        {
            ee.entry[i].ndimParray = 2;
            ee.entry[i].Parray = new double[ee.entry[i].ndimParray];
            ee.entry[i].ndimGarray = 4;
            ee.entry[i].Garray = new double[ee.entry[i].ndimGarray];
        }

        delete[] tmp;
        tmp = new double *[3];

        for (int i = 0; i < ee.entries; i++)
        {
            tmp[0] = ee.entry[i].Parray;
            tmp[1] = &ee.entry[i].Fvalue;
            tmp[2] = ee.entry[i].Garray;

            int nCols[3];
            nCols[0] = ee.entry[i].ndimParray;
            nCols[1] = 1;
            nCols[2] = ee.entry[i].ndimGarray;

            f.loadMatrix<double>(tmp, 3, nCols, 1);
        }

        for (int i = 0; i < dd.entries; i++)
        {
            for (int j = 0; j < dd.entry[i].ndimParray; j++)
            {
                EXPECT_DOUBLE_EQ(ee.entry[i].Parray[j], dd.entry[i].Parray[j]);
            }
            EXPECT_DOUBLE_EQ(ee.entry[i].Fvalue, dd.entry[i].Fvalue);
            for (int j = 0; j < dd.entry[i].ndimGarray; j++)
            {
                EXPECT_DOUBLE_EQ(ee.entry[i].Garray[j], dd.entry[i].Garray[j]);
            }
        }

        f.closeFile();
        // delete the file
        std::remove(fileName);

        delete[] tmp;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
