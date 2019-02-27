#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "app.h"

using namespace flappy;

int main() {
  App app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
