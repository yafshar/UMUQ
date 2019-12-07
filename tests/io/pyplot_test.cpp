#include "io/pyplot.hpp"
#include "gtest/gtest.h"

#ifdef HAVE_PYTHON

using namespace umuq;

/*!
 * \ingroup Global_Module
 *
 * \brief Create a global instance of the Pyplot from Pyplot library
 *
 */
umuq::matplotlib_223::pyplot plt;

// TEST for Basic functionality
TEST(Pyplot_test, HandlesBasic)
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

    std::cout << plt.get_backend() << std::endl;

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

    // close figure
    EXPECT_TRUE(plt.close());
}

// TEST for fill_between functionality
TEST(Pyplot_test, HandlesFill_Between)
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

    // Plot line from given x and y data. Color is selected automatically.
    EXPECT_TRUE(plt.plot<double>(x, y, "b:", "cos(x)"));

    // Plot a red dashed line from given x and z data and show up as "log(x)" in the legend
    EXPECT_TRUE(plt.plot<double>(x, z, "r--", "sin(x) + cos(x)"));

    // save figure
    EXPECT_TRUE(plt.fill_between<double>(x, y, z, keywords));

    // Enable legend.
    EXPECT_TRUE(plt.legend());

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // close figure
    EXPECT_TRUE(plt.close());
}

// // TEST for animation functionality
// TEST(Pyplot_test, HandlesAnimation)
// {
//     // Prepare data.
//     int n = 50;

//     std::vector<double> x(n);
//     std::vector<double> y(n);
//     std::vector<double> z(n);

//     double const dx = 4 * M_PI / (n - 1);
//     double t(0);

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
//
//     // close figure
//     EXPECT_TRUE(plt.close());
// }

// TEST for histogram functionality
TEST(Pyplot_test, HandlesHist)
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

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Set the size of output image = 1200x780 pixels
    EXPECT_TRUE(plt.figure(1200, 780));

    // histogram figure
    EXPECT_TRUE(plt.hist<double>(x, 50, true, "", "Histogram", 1.0, 51, 255, 51));

    // Add graph title
    EXPECT_TRUE(plt.title("Histogram"));

    // Enable legend.
    EXPECT_TRUE(plt.legend());

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // close figure
    EXPECT_TRUE(plt.close());
}

// TEST for scatter functionality
TEST(Pyplot_test, HandlesScatterWithArrayOfColors)
{
    std::string fileName = "./scatter.svg";
    std::remove(fileName.c_str());

    // Prepare data.
    int n = 100;

    // X coordinates
    std::vector<double> x(n);
    // Y coordinates
    std::vector<double> y(n);
    // Marker size in points**2
    std::vector<int> s(1, 1000);
    // // Scalar data color
    std::vector<double> c(n);

    // std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0, 1.0);
    std::uniform_int_distribution<> idis(250, 255);

    std::for_each(x.begin(), x.end(), [&](double &x_i) { x_i = dis(gen); });
    std::for_each(y.begin(), y.end(), [&](double &y_i) { y_i = dis(gen); });
    std::for_each(c.begin(), c.end(), [&](double &c_i) { c_i = idis(gen); });

    // Prepare keywords to pass to PolyCollection. See
    std::map<std::string, std::string> keywords;
    keywords["marker"] = "o";

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Set the size of output image = 1200x780 pixels
    EXPECT_TRUE(plt.figure(1200, 780));

    // Create scatter plot
    EXPECT_TRUE(plt.scatter<double>(x, y, s, c, keywords));

    // Add graph title
    EXPECT_TRUE(plt.title("Scatter"));

    // Enable legend.
    EXPECT_TRUE(plt.legend());

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // close figure
    EXPECT_TRUE(plt.close());
}

TEST(Pyplot_test, HandlesScatterWithFormat)
{
    std::string fileName = "./scatter.png";
    std::remove(fileName.c_str());

    // Prepare data.
    int n = 11 * 11;

    // X coordinates
    std::vector<double> x(n);
    // Y coordinates
    std::vector<double> y(n);

    double dx = 0.1;
    double dy = 0.1;

    for (int i = 0, k = 0; i < 11; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            x[k] = i * dx;
            y[k] = j * dy;
            k++;
        }
    }

    // Prepare keywords to pass to PolyCollection. See
    std::map<std::string, std::string> keywords;
    keywords["marker"] = "D";

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Set the size of output image = 1200x780 pixels
    EXPECT_TRUE(plt.figure(1200, 780));

    // Create scatter plot
    EXPECT_TRUE(plt.scatter<double>(x, y, 200, "b", keywords));

    // Add graph title
    EXPECT_TRUE(plt.title("Scatter"));

    // Enable legend.
    EXPECT_TRUE(plt.legend());

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // close figure
    EXPECT_TRUE(plt.close());
}

// TEST for contour functionality
TEST(Pyplot_test, HandlesContour)
{
    std::string fileName = "./contour.png";
    std::remove(fileName.c_str());

    // Prepare data.
    int nDimX = 20;
    int nDimY = 8;

    // X coordinates
    std::vector<double> x(nDimX);
    std::iota(x.begin(), x.end(), 1.0);

    // Y coordinates
    std::vector<double> y(nDimY);
    std::iota(y.begin(), y.end(), 1.0);

    // The height values over which the contour is drawn
    std::vector<double> z(nDimX * nDimY);

    // std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(0, 1.0);

    std::for_each(z.begin(), z.end(), [&](double &z_i) { z_i = dis(gen); });

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Set the size of output image = 2000x800 pixels
    EXPECT_TRUE(plt.figure(2000, 800));

    // Create scatter plot
    EXPECT_TRUE(plt.contour<double>(x, y, z));

    // Add graph title
    EXPECT_TRUE(plt.title("Contour"));

    // Enable legend.
    EXPECT_TRUE(plt.legend());

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // close figure
    EXPECT_TRUE(plt.close());
}

// TEST for contour functionality
TEST(Pyplot_test, HandlesContourf)
{
    std::string fileName = "./contourf.png";
    std::remove(fileName.c_str());

    // Prepare data.
    int nDimX = 20;
    int nDimY = 8;

    // X coordinates
    std::vector<double> x(nDimX);
    std::iota(x.begin(), x.end(), 1.0);

    // Y coordinates
    std::vector<double> y(nDimY);
    std::iota(y.begin(), y.end(), 1.0);

    // The height values over which the contour is drawn
    std::vector<double> z(nDimX * nDimY);

    // std::random_device rd;
    std::mt19937 gen(123);
    std::exponential_distribution<> dis(1.0);

    std::for_each(z.begin(), z.end(), [&](double &z_i) { z_i = dis(gen); });

    // Clear previous plot
    EXPECT_TRUE(plt.clf());

    // Set the size of output image = 2000x800 pixels
    EXPECT_TRUE(plt.figure(2000, 800));

    // Create scatter plot
    EXPECT_TRUE(plt.contourf<double>(x, y, z));

    // Add graph title
    EXPECT_TRUE(plt.title("Contourf"));

    // Enable legend.
    EXPECT_TRUE(plt.legend());

    // save figure
    EXPECT_TRUE(plt.savefig(fileName));

    // close figure
    EXPECT_TRUE(plt.close());
}

#else

// TEST for Basic functionality
TEST(Pyplot_test, HandlesBasic)
{
}

#endif //HAVE_PYTHON

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
