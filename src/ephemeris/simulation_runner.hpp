#pragma once

#include "ephemeris/simulation.hpp"
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace sf::ephemeris {

struct SimulationPublication {
    SystemSnapshot snapshot;
    SimulationClock clock;
    CommandResult result;
    double advance_ms = 0.0;
    double copy_ms = 0.0;
    double nbody_epsilon = 0.0;
};

class SimulationRunner {
public:
    ~SimulationRunner();
    void set_threaded(bool enabled);
    bool threaded() const noexcept { return worker_.joinable(); }
    const SystemSnapshot& snapshot() const noexcept { return threaded() ? front_.snapshot : simulation_.snapshot(); }
    const SimulationClock& clock() const noexcept { return threaded() ? front_.clock : simulation_.clock(); }
    double nbody_epsilon() const noexcept { return threaded() ? front_.nbody_epsilon : simulation_.nbody_epsilon(); }
    const SimulationPublication& publication() const noexcept { return front_; }

    CommandResult replace_system(LoadedSystem loaded);
    template<class Command> CommandResult apply(Command command) {
        return execute([command = std::move(command)](Simulation& simulation) { return simulation.apply(command); });
    }
    CommandResult advance(units::Seconds elapsed);
    CommandResult advance_async(units::Seconds elapsed);
    CommandResult poll();
    CommandResult wait();

private:
    CommandResult execute(std::function<CommandResult(Simulation&)> command);
    CommandResult submit(units::Seconds elapsed);
    void adopt_locked();
    void work();
    void stop() noexcept;

    Simulation simulation_;
    SimulationPublication front_, ready_, back_;
    std::mutex mutex_;
    std::condition_variable work_ready_, completed_;
    std::thread worker_;
    std::function<CommandResult(Simulation&)> command_;
    CommandResult command_result_;
    std::exception_ptr failure_;
    double pending_seconds_ = 0.0;
    uint64_t submitted_id_ = 0, completed_id_ = 0;
    bool pending_ = false, ready_available_ = false, stopping_ = false;
};

}
