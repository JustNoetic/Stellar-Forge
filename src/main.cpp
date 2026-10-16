#include "app.hpp"
#include <cstring>

int main(int argc, char** argv) {
    bool self_test = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--self-test") == 0) {
            self_test = true;
        }
    }

    sf::App app;
    app.init(self_test);
    app.run();
    app.shutdown();
    return 0;
}
