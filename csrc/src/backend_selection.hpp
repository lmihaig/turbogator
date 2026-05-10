#pragma once

#include "fake_backend.hpp"

// backend implementation to use.
using ActiveBackend = FakeBackend;

inline ActiveBackend create_backend() {
    return ActiveBackend{};
}
