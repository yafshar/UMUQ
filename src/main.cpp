#include "environment.hpp"

int main(int argc, char **argv)
{
    // Create a torc environment object
    umuq::Torc.reset(new umuq::torcEnvironment);

    umuq::Torc->SetUp();

    umuq::Torc->TearDown();

    return 0;
}
