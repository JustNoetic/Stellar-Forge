#include "app.hpp"
#include "core/log.hpp"
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        auto options = sf::benchmark::parse_options(argc, argv);
        if (options.help) {
            std::cout << sf::benchmark::help();
            return 0;
        }
        sf::App app;
        app.init(options.self_test, std::move(options.benchmark));
        try {
            app.run();
        } catch (...) {
            app.shutdown();
            throw;
        }
        app.shutdown();
        return 0;
    } catch (const std::exception& error) {
        sf::log::error("{}", error.what());
        return 1;
    }
}
