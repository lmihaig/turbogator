#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "backends/ops.hpp"

namespace py = pybind11;

namespace {

py::array_t<float> make_out_like(const py::array_t<float>& in) {
    return py::array_t<float>(in.size());
}

}

PYBIND11_MODULE(turbogator_ext, m) {
    m.def("geometric_product_baseline", [](py::array_t<float> a, py::array_t<float> b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("geometric_product_baseline: size mismatch");
        }
        auto out = make_out_like(a);
        tg::geometric_product_baseline(a.data(), b.data(), out.mutable_data(), a.size());
        return out;
    });

    m.def("geometric_product_vectorized", [](py::array_t<float> a, py::array_t<float> b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("geometric_product_vectorized: size mismatch");
        }
        auto out = make_out_like(a);
        tg::geometric_product_vectorized(a.data(), b.data(), out.mutable_data(), a.size());
        return out;
    });

    m.def("equi_join_baseline", [](py::array_t<float> a, py::array_t<float> b, py::array_t<float> ref) {
        if (a.size() != b.size()) {
            throw std::runtime_error("equi_join_baseline: size mismatch");
        }
        auto out = make_out_like(a);
        tg::equi_join_baseline(a.data(), b.data(), ref.size() ? ref.data() : nullptr, out.mutable_data(), a.size());
        return out;
    });

    m.def("equi_geometric_attention_baseline", [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v) {
        if (q.size() != k.size() || q.size() != v.size()) {
            throw std::runtime_error("equi_geometric_attention_baseline: size mismatch");
        }
        auto out = make_out_like(q);
        tg::equi_geometric_attention_baseline(q.data(), k.data(), v.data(), out.mutable_data(), q.size());
        return out;
    });

    m.def("scaler_gated_gelu_baseline", [](py::array_t<float> x) {
        auto out = make_out_like(x);
        tg::scaler_gated_gelu_baseline(x.data(), out.mutable_data(), x.size());
        return out;
    });
}
