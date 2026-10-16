#include "ephemeris/simulation_runner.hpp"
#include "core/timing.hpp"
#include <cmath>

namespace sf::ephemeris {

SimulationRunner::~SimulationRunner() { stop(); }

void SimulationRunner::set_threaded(bool enabled) {
    if (enabled == threaded()) return;
    if (!enabled) {
        wait();
        stop();
        return;
    }
    front_ = {simulation_.snapshot(), simulation_.clock(), {}, 0.0, 0.0, simulation_.nbody_epsilon()};
    ready_ = front_;
    back_ = front_;
    stopping_ = false;
    failure_ = {};
    worker_ = std::thread([this] { work(); });
}

void SimulationRunner::stop() noexcept {
    if (!threaded()) return;
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    work_ready_.notify_one();
    worker_.join();
}

CommandResult SimulationRunner::replace_system(LoadedSystem loaded) {
    return execute([loaded = std::move(loaded)](Simulation& simulation) mutable {
        return simulation.replace_system(std::move(loaded));
    });
}

CommandResult SimulationRunner::execute(std::function<CommandResult(Simulation&)> command) {
    if (!threaded()) return command(simulation_);
    wait();
    {
        std::lock_guard lock(mutex_);
        command_ = std::move(command);
        ++submitted_id_;
    }
    work_ready_.notify_one();
    wait();
    return command_result_;
}

CommandResult SimulationRunner::submit(units::Seconds elapsed) {
    if (!std::isfinite(elapsed.value) || elapsed.value < 0.0)
        return {false, false, "Frame time must be finite and nonnegative."};
    if (elapsed.value == 0.0 || clock().paused || snapshot().empty()) return {};
    {
        std::lock_guard lock(mutex_);
        if (failure_) std::rethrow_exception(failure_);
        if (!std::isfinite(pending_seconds_ + elapsed.value))
            return {false, false, "Pending simulation time overflow."};
        pending_seconds_ += elapsed.value;
        pending_ = true;
        ++submitted_id_;
    }
    work_ready_.notify_one();
    return {};
}

CommandResult SimulationRunner::advance(units::Seconds elapsed) {
    if (!threaded()) return simulation_.advance(elapsed);
    auto result = submit(elapsed);
    return result ? wait() : result;
}

CommandResult SimulationRunner::advance_async(units::Seconds elapsed) {
    if (!threaded()) return simulation_.advance(elapsed);
    auto result = poll();
    return result ? submit(elapsed) : result;
}

void SimulationRunner::adopt_locked() {
    if (failure_) std::rethrow_exception(failure_);
    if (ready_available_) {
        std::swap(front_, ready_);
        ready_available_ = false;
    }
}

CommandResult SimulationRunner::poll() {
    if (!threaded()) return {};
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (lock.owns_lock()) adopt_locked();
    return front_.result;
}

CommandResult SimulationRunner::wait() {
    if (!threaded()) return {};
    std::unique_lock lock(mutex_);
    auto target = submitted_id_;
    completed_.wait(lock, [&] { return failure_ || completed_id_ >= target; });
    adopt_locked();
    return front_.result;
}

void SimulationRunner::work() {
    try {
        CommandResult advance_result;
        for (;;) {
            std::unique_lock lock(mutex_);
            work_ready_.wait(lock, [&] { return stopping_ || pending_ || command_; });
            if (stopping_) return;
            auto command = std::exchange(command_, {});
            auto elapsed = units::Seconds{pending_seconds_};
            auto id = submitted_id_;
            pending_seconds_ = 0.0;
            pending_ = false;
            lock.unlock();

            auto start = SteadyClock::now();
            auto result = command ? command(simulation_) : advance_result ? simulation_.advance(elapsed) : advance_result;
            back_.advance_ms = elapsed_ms(start);
            if (!result && !command) simulation_.apply(SetPaused{true});
            if (!command || result) advance_result = command ? CommandResult{} : result;
            start = SteadyClock::now();
            back_.snapshot = simulation_.snapshot();
            back_.clock = simulation_.clock();
            back_.nbody_epsilon = simulation_.nbody_epsilon();
            back_.result = advance_result;
            back_.copy_ms = elapsed_ms(start);

            lock.lock();
            std::swap(back_, ready_);
            ready_available_ = true;
            if (command) command_result_ = result;
            completed_id_ = id;
            lock.unlock();
            completed_.notify_one();
        }
    } catch (...) {
        {
            std::lock_guard lock(mutex_);
            failure_ = std::current_exception();
        }
        completed_.notify_one();
    }
}

}
