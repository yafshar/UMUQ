#include "core/core.hpp"
#include "io/matplotlib.hpp"
#include "gtest/gtest.h"

#ifdef HAVE_PYTHON

//! TEST for Basic functionality
TEST(Matplotlib_test, HandlesBasic)
{
    std::string fileName = "./basic.png";
    std::remove(fileName.c_str());

    // Prepare data.
    int n = 5000;

    std::vector<double> x(n);
    std::vector<double> y(n);
    std::vector<double> z(n);

    double const dx = 4 * M_PI / (n - 1);
    double t(0);

    for (int i = 0; i < n; ++i)
    {
        x[i] = t;
        y[i] = std::cos(t);
        z[i] = std::sin(t) + std::cos(t);
        t += dx;
    }

    //Create an instance of the Pyplot from Matplotlib library
    pyplot plt;

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Set the size of output image = 1200x780 pixels
    EXPECT_TRUE(plt.figure(1200, 780));

    // Plot line from given x and y data. Color is selected automatically.
    EXPECT_TRUE(plt.plot<double>(x, y, "", "cos(x)"));

    // Plot a red dashed line from given x and z data and show up as "log(x)" in the legend
    EXPECT_TRUE(plt.plot<double>(x, z, "r--", "sin(x) + cos(x)"));

    // Set x-axis to interval \f$ \[0,4\pi\] \f$
    EXPECT_TRUE(plt.xlim<double>(0., 4 * M_PI));

    // Add graph title
    EXPECT_TRUE(plt.title("Basic functionality"));

    // Enable legend.
    EXPECT_TRUE(plt.legend());

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // Delete the file
    std::remove(fileName.c_str());
}

//! TEST for fill_between functionality
TEST(Matplotlib_test, HandlesFill_Between)
{
    std::string fileName = "./fill_between.png";
    std::remove(fileName.c_str());

    // Prepare data.
    int n = 5000;

    std::vector<double> x(n);
    std::vector<double> y(n);
    std::vector<double> z(n);

    double const dx = 4 * M_PI / (n - 1);
    double t(0);

    for (int i = 0; i < n; ++i)
    {
        x[i] = t;
        y[i] = std::cos(t);
        z[i] = std::sin(t) + std::cos(t);
        t += dx;
    }

    // Create an instance of the Pyplot from Matplotlib library
    pyplot plt;

    // Prepare keywords to pass to PolyCollection. See
    std::map<std::string, std::string> keywords;
    keywords["alpha"] = "0.4";
    keywords["color"] = "grey";
    keywords["hatch"] = "-";

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Set the size of output image = 1200x780 pixels
    EXPECT_TRUE(plt.figure(1200, 780));

    // Add graph title
    EXPECT_TRUE(plt.title("Fill_between"));

    // save figure
    EXPECT_TRUE(plt.fill_between<double>(x, y, z, keywords));

    // Enable legend.
    EXPECT_TRUE(plt.legend());

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // Delete the file
    std::remove(fileName.c_str());
}

// //! TEST for animation functionality
// TEST(Matplotlib_test, HandlesAnimation)
// {
//     // Prepare data.
//     int n = 50;

//     std::vector<double> x(n);
//     std::vector<double> y(n);
//     std::vector<double> z(n);

//     double const dx = 4 * M_PI / (n - 1);
//     double t(0);

//     // Create an instance of the Pyplot from Matplotlib library
//     pyplot plt;

//     for (int i = 0; i < n; ++i)
//     {
//         x.push_back(t);
//         y.push_back(std::cos(t));
//         z.push_back(std::sin(t) + std::cos(t));

//         if (i % 5 == 0)
//         {
//             // Clear previous plot
//             EXPECT_TRUE(plt.clf());

//             // Plot line from given x and y data. Color is selected automatically.
//             EXPECT_TRUE(plt.plot<double>(x, y, "", "cos(x)"));

//             // Plot a line whose name will show up as "sin(x) + cos(x)" in the legend.
//             EXPECT_TRUE(plt.plot<double>(x, z, "r--", "sin(x) + cos(x)"));

//             // Set x-axis to interval \f$ \[0,4\pi\] \f$
//             EXPECT_TRUE(plt.xlim<double>(0, 4 * M_PI));

//             // Set y-axis to interval \f$ \[-2,2\] \f$
//             EXPECT_TRUE(plt.ylim<double>(-2.0, 2.0));

//             // Add graph title
//             EXPECT_TRUE(plt.title("Animation figure"));

//             // Enable legend.
//             EXPECT_TRUE(plt.legend());

//             // Display plot continuously
//             EXPECT_TRUE(plt.draw());

//             EXPECT_TRUE(plt.pause(0.0001));
//         }
//         t += dx;
//     }
// }

//! TEST for histogram functionality
TEST(Matplotlib_test2, HandlesHist)
{
    std::string fileName = "./hist.png";
    std::remove(fileName.c_str());

    // Prepare data.
    int n = 10000;

    std::vector<double> x(n);

    // std::random_device rd;
    std::mt19937 gen(123);
    std::normal_distribution<> d;

    std::for_each(x.begin(), x.end(), [&](double &x_i) { x_i = d(gen); });

    // Create an instance of the Pyplot from Matplotlib library
    pyplot plt;

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Set the size of output image = 1200x780 pixels
    EXPECT_TRUE(plt.figure(1200, 780));

    // save figure
    EXPECT_TRUE(plt.hist<double>(x, 50, true, "r", "Histogram", 0.5));

    // Add graph title
    EXPECT_TRUE(plt.title("Histogram"));

    // Enable legend.
    EXPECT_TRUE(plt.legend());

    // // Set x-axis to interval \f$ \[0,9\] \f$
    // EXPECT_TRUE(plt.xlim<double>(-1., 11.));

    // Set y-axis to interval \f$ \[0,1\] \f$
    // EXPECT_TRUE(plt.ylim<double>(0, 400.));

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // Delete the file
    std::remove(fileName.c_str());
}

#else
TEST(Matplotlib_test, HandlesBasic)
{
}
#endif //HAVE_PYTHON

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}