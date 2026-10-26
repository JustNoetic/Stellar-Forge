#include "ephemeris/simulation_runner.hpp"
#include "core/timing.hpp"
#include <cmath>

namespace sf::ephemeris {

SimulationRunner::~SimulationRunner() {
    cancel_timeline_render();
    stop();
}

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

void SimulationRunner::swap_simulation(SimulationRunner& other) {
    if (&other == this) return;
    bool threaded_here = threaded(), threaded_there = other.threaded();
    set_threaded(false);
    other.set_threaded(false);
    std::swap(simulation_, other.simulation_);
    set_threaded(threaded_here);
    other.set_threaded(threaded_there);
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

TimelineRenderProgress SimulationRunner::timeline_progress() const {
    TimelineRenderProgress p;
    p.active = timeline_active_.load(std::memory_order_relaxed);
    p.progress = timeline_progress_.load(std::memory_order_relaxed);
    p.steps_done = timeline_steps_done_.load(std::memory_order_relaxed);
    p.steps_total = timeline_steps_total_.load(std::memory_order_relaxed);
    p.sim_rate_yr_s = timeline_sim_rate_.load(std::memory_order_relaxed);
    p.completed = timeline_completed_.load(std::memory_order_relaxed);
    p.cancelled = timeline_cancel_.load(std::memory_order_relaxed);
    {
        std::lock_guard lk(timeline_mutex_);
        p.error = timeline_error_;
    }
    return p;
}

void SimulationRunner::cancel_timeline_render() {
    timeline_cancel_ = true;
    if (timeline_worker_.joinable()) {
        timeline_worker_.join();
    }
    timeline_active_ = false;
}

std::shared_ptr<TimelineRecording> SimulationRunner::take_completed_timeline() {
    std::lock_guard lk(timeline_mutex_);
    timeline_completed_ = false;
    return std::move(completed_timeline_);
}

void SimulationRunner::apply_timeline_frame(const TimelineRecording& rec, size_t frame_index) {
    execute([&](Simulation& sim) {
        sim.apply_timeline_frame(rec, frame_index);
        return CommandResult{true, true, {}};
    });
}

void SimulationRunner::start_timeline_render(units::Years target_time, size_t requested_steps) {
    cancel_timeline_render();
    apply(SetPaused{true});

    SystemDefinition def;
    std::vector<BodyMotion> motions;
    JacobiHierarchy jacobi;
    units::Years clock_time;
    Ias15Integrator ias15;
    JacobiPropagator kepler;
    std::shared_ptr<SpiceEphemeris> spice;
    double epoch_et = 0.0;
    double p_min = 1.0;

    execute([&](Simulation& sim) {
        def = sim.snapshot().definition();
        motions.assign(sim.snapshot().motions().begin(), sim.snapshot().motions().end());
        jacobi = sim.snapshot().jacobi_hierarchy();
        clock_time = sim.clock().time;
        ias15 = sim.ias15();
        kepler = sim.kepler();
        spice = sim.spice();
        epoch_et = sim.epoch_et();
        p_min = sim.estimate_min_orbital_period();
        return CommandResult{true, false, {}};
    });

    timeline_active_ = true;
    timeline_cancel_ = false;
    timeline_completed_ = false;
    timeline_progress_ = 0.0f;
    timeline_steps_done_ = 0;
    timeline_steps_total_ = 0;
    timeline_sim_rate_ = 0.0;
    {
        std::lock_guard lk(timeline_mutex_);
        timeline_error_.clear();
        completed_timeline_.reset();
    }

    timeline_worker_ = std::thread([this, def = std::move(def), motions = std::move(motions),
                                    jacobi = std::move(jacobi), clock_time, ias15, kepler,
                                    spice, epoch_et, p_min, target_time, requested_steps]() mutable {
        double diff_t = target_time.value - clock_time.value;
        if (std::abs(diff_t) < 1e-9) {
            std::lock_guard lk(timeline_mutex_);
            timeline_error_ = "Target time is identical to current time.";
            timeline_active_ = false;
            return;
        }

        size_t num_steps = requested_steps;
        if (num_steps == 0) {
            num_steps = static_cast<size_t>(std::ceil(std::abs(diff_t) / p_min * 64.0));
            num_steps = std::clamp(num_steps, size_t(1000), size_t(200000));
        }
        timeline_steps_total_ = num_steps;
        double step_dt = diff_t / static_cast<double>(num_steps);

        auto recording = std::make_shared<TimelineRecording>();
        recording->start_time = clock_time;
        recording->end_time = target_time;
        recording->source_model = def.motion_model;
        recording->times.resize(num_steps);
        const size_t num_bodies = def.bodies.size();
        recording->motions.resize(num_steps * num_bodies);

        auto current_motions = motions;
        auto start_wall = SteadyClock::now();

        for (size_t step = 0; step < num_steps; ++step) {
            if (timeline_cancel_.load(std::memory_order_relaxed)) {
                timeline_active_ = false;
                return;
            }

            units::Years t_k{clock_time.value + (step + 1) * step_dt};

            if (def.motion_model == MotionModel::NBody) {
                auto res = ias15.integrate(def, current_motions, t_k, 0, &timeline_cancel_);
                if (!res && !res.changed) {
                    std::lock_guard lk(timeline_mutex_);
                    timeline_error_ = res.error;
                    timeline_active_ = false;
                    return;
                }
            } else if (def.motion_model == MotionModel::Keplerian) {
                kepler.evaluate(def, jacobi, t_k, current_motions);
            } else if (def.motion_model == MotionModel::Spice) {
                if (!spice) {
                    std::lock_guard lk(timeline_mutex_);
                    timeline_error_ = "SPICE runtime is not loaded.";
                    timeline_active_ = false;
                    return;
                }
                double et_k = epoch_et + units::to_seconds(t_k).value;
                for (size_t b = 0; b < num_bodies; ++b) {
                    auto st = spice->state(def.bodies[b].spice_id, et_k, current_motions[b]);
                    if (!st && !def.bodies[b].is_spacecraft) {
                        std::lock_guard lk(timeline_mutex_);
                        timeline_error_ = def.bodies[b].name + ": " + st.error;
                        timeline_active_ = false;
                        return;
                    }
                }
            }

            recording->times[step] = t_k;
            std::copy(current_motions.begin(), current_motions.end(), recording->motions.begin() + step * num_bodies);

            if ((step & 63) == 0 || step + 1 == num_steps) {
                timeline_steps_done_ = step + 1;
                timeline_progress_ = static_cast<float>(step + 1) / static_cast<float>(num_steps);
                double elapsed_sec = elapsed_ms(start_wall) / 1000.0;
                if (elapsed_sec > 0.01) {
                    timeline_sim_rate_ = std::abs((step + 1) * step_dt) / elapsed_sec;
                }
            }
        }

        {
            std::lock_guard lk(timeline_mutex_);
            completed_timeline_ = std::move(recording);
        }
        timeline_completed_ = true;
        timeline_active_ = false;
    });
}

}
