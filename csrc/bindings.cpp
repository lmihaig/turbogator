#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "backends/ops.hpp"

namespace py = pybind11;

namespace {

py::array_t<float> make_out_like(const py::array_t<float>& in) {
    return py::array_t<float>(in.size());
}

py::array_t<float> make_out_with_shape(const py::buffer_info& info, const std::vector<ssize_t>& shape) {
    return py::array_t<float>(shape, info.strides);
}

py::array_t<float> make_equi_linear_out(const py::array_t<float>& x, const py::array_t<float>& weight) {
    auto x_info = x.request();
    auto w_info = weight.request();

    if (x_info.shape.size() < 2 || w_info.shape.size() < 2) {
        return py::array_t<float>(x.size());
    }

    const ssize_t out_channels = w_info.shape[0];
    std::vector<ssize_t> shape = x_info.shape;
    shape[shape.size() - 2] = out_channels;
    return py::array_t<float>(shape);
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

    m.def("equi_linear_baseline", [](py::array_t<float> x, py::array_t<float> weight, py::array_t<float> bias, bool normalize_basis) {
        (void)normalize_basis;
        auto out = make_equi_linear_out(x, weight);
        const float* bias_ptr = bias.size() ? bias.data() : nullptr;
        tg::equi_linear_baseline(x.data(), weight.data(), bias_ptr, out.mutable_data(), out.size());
        return out;
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("normalize_basis") = true);

    m.def("equi_rms_norm_baseline", [](py::array_t<float> x, py::array_t<float> weight, float eps) {
        (void)eps;
        auto out = make_out_like(x);
        const float* weight_ptr = weight.size() ? weight.data() : nullptr;
        tg::equi_rms_norm_baseline(x.data(), weight_ptr, out.mutable_data(), out.size());
        return out;
    }, py::arg("x"), py::arg("weight"), py::arg("eps") = 0.0f);
}
